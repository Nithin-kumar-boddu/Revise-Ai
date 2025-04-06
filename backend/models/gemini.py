import google.generativeai as genai
import os
import time
import json
import re
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GeminiAPI")

class GeminiAPI:
    """Wrapper class for Google's Gemini API with industry-level enhancements for accuracy and reliability"""

    def __init__(
        self, 
        api_key=None, 
        model_name="models/gemini-2.0-flash",
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        safety_settings=None,
        filter_research_sections=True,
        filter_conclusion_remarks=True
    ):
        """
        Initialize the Gemini API with the provided API key, model and generation parameters.
        
        Args:
            api_key (str): The API key for accessing Google's Gemini API
            model_name (str): The specific model to use
            temperature (float): Controls randomness (0.0-1.0)
            top_p (float): Nucleus sampling parameter (0.0-1.0)
            top_k (int): Limits token selection to top k options
            max_output_tokens (int): Maximum length of generated content
            safety_settings (dict, optional): Custom safety settings
            filter_research_sections (bool): Whether to filter out "ongoing research" sections
            filter_conclusion_remarks (bool): Whether to filter out concluding remarks
        """
        if not api_key:
            # Try to get API key from environment variable if not provided
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("API key is required for Gemini API. Provide as parameter or set GEMINI_API_KEY environment variable.")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        try:
            # Validate the model
            available_models = self.list_available_models()
            if model_name not in available_models:
                logger.warning(f"Model '{model_name}' not found in available models. Available models: {available_models}")
                # Fall back to the most capable available model if requested one isn't available
                model_name = next((m for m in available_models if "gemini" in m.lower() and "pro" in m.lower()), available_models[0])
                logger.info(f"Falling back to model: {model_name}")
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            logger.info("Proceeding with requested model without validation.")
        
        # Default safety settings if none provided
        if safety_settings is None:
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
            }
            
        # Store generation parameters
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        # Initialize model with generation config and safety settings
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=safety_settings
        )
        
        self.model_name = model_name
        logger.info(f"Initialized GeminiAPI with model: {model_name}")
        
        # Store request history for debugging and optimization
        self.request_history = []
        
        # Content filtering options
        self.filter_research_sections = filter_research_sections
        self.filter_conclusion_remarks = filter_conclusion_remarks
        
    @staticmethod
    def list_available_models():
        """Retrieve and return a list of available model names."""
        try:
            models = genai.list_models()
            return [model.name for model in models]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
            
    def _handle_rate_limits(self, retry_attempt=0, max_retries=3):
        """Handle rate limiting with exponential backoff."""
        if retry_attempt >= max_retries:
            raise Exception(f"Maximum retry attempts ({max_retries}) exceeded")
        
        wait_time = 2 ** retry_attempt  # Exponential backoff
        logger.info(f"Rate limit hit, waiting {wait_time} seconds before retry")
        time.sleep(wait_time)
        return retry_attempt + 1
        
    def _log_request(self, prompt, response, metadata=None):
        """Log request and response for analysis and improvement."""
        request_data = {
            "timestamp": time.time(),
            "prompt": prompt,
            "model": self.model_name,
            "config": self.generation_config,
            "metadata": metadata or {}
        }
        
        if hasattr(response, "candidates"):
            request_data["response"] = {
                "text": response.text,
                "candidates": len(response.candidates),
                "finish_reason": response.candidates[0].finish_reason if response.candidates else None
            }
        else:
            request_data["response"] = {"text": str(response)}
            
        self.request_history.append(request_data)
        
        # Log only a summary to avoid excessive output
        logger.debug(f"Request logged: {len(str(prompt))} chars -> {len(request_data['response']['text']) if 'text' in request_data['response'] else 0} chars response")
        
    def _filter_content(self, text):
        """
        Filter out unwanted sections from the generated content.
        
        Args:
            text (str): The text to filter
            
        Returns:
            str: The filtered text
        """
        if not text:
            return text
            
        # Filter out "Areas of Ongoing Research" or similar sections
        if self.filter_research_sections:
            # Look for headings related to research or debate
            patterns = [
                r"(?i)#+ *(?:areas? of )?(?:ongoing|current) research.*?(?:\n#|\Z)",
                r"(?i)#+ *(?:areas? of )?debate.*?(?:\n#|\Z)",
                r"(?i)#+ *(?:future|advanced) directions.*?(?:\n#|\Z)",
                r"(?i)#+ *(?:less relevant|beyond the scope).*?(?:\n#|\Z)",
                r"(?i)^[0-9]+\. *(?:areas? of )?(?:ongoing|current) research.*?(?:\n[0-9]+\.|\Z)",
                r"(?i)^[0-9]+\. *(?:areas? of )?debate.*?(?:\n[0-9]+\.|\Z)",
                r"(?i)^[0-9]+\. *(?:future|advanced) directions.*?(?:\n[0-9]+\.|\Z)",
                r"(?i)^[0-9]+\. *(?:less relevant|beyond the scope).*?(?:\n[0-9]+\.|\Z)"
            ]
            
            for pattern in patterns:
                text = re.sub(pattern, "", text, flags=re.DOTALL)
        
        # Filter out concluding remarks
        if self.filter_conclusion_remarks:
            # Look for concluding paragraphs with phrases like "good luck with your exam"
            patterns = [
                r"(?i)(?:this explanation should|hope this helps|this guide should).*?(?:good luck|best of luck|all the best).*?(?:exam|test|assessment).*?$",
                r"(?i)(?:remember to practice|practice .*? to solidify).*?$",
                r"(?i)(?:good luck|best of luck|all the best).*?(?:exam|test|assessment).*?$"
            ]
            
            for pattern in patterns:
                text = re.sub(pattern, "", text, flags=re.DOTALL)
        
        # Clean up any excess newlines from the filtering
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        
        return text
        
    def _create_structured_prompt(self, content, system_prompt=None, format_instructions=None):
        """
        Create a structured prompt compatible with Gemini API.
        
        Args:
            content (str or list): The main content
            system_prompt (str, optional): System instructions
            format_instructions (str, optional): Format guidelines
            
        Returns:
            The properly formatted prompt for Gemini API
        """
        # For simple string content
        if isinstance(content, str):
            main_content = content
            if format_instructions:
                main_content = f"{main_content}\n\n{format_instructions}"
                
            # If system prompt is provided, we need to format as a chat
            if system_prompt:
                return [
                    {"role": "user", "parts": [{"text": system_prompt}]},
                    {"role": "model", "parts": [{"text": "I'll help you with that."}]},
                    {"role": "user", "parts": [{"text": main_content}]}
                ]
            else:
                # For simple prompt, just return the content directly
                return main_content
        
        # For already structured content (assuming it's already in correct format)
        elif isinstance(content, list):
            # If we have format instructions, add them to the last user message
            if format_instructions:
                for i in reversed(range(len(content))):
                    if content[i].get("role") == "user":
                        # Find the last part with text
                        for j in reversed(range(len(content[i].get("parts", [])))):
                            part = content[i]["parts"][j]
                            if isinstance(part, dict) and "text" in part:
                                content[i]["parts"][j]["text"] += f"\n\n{format_instructions}"
                                break
                        break
            
            # If we have a system prompt and the first message isn't from system
            if system_prompt and (not content or content[0].get("role") != "user"):
                content.insert(0, {"role": "user", "parts": [{"text": system_prompt}]})
                # Add a model response to the system prompt
                content.insert(1, {"role": "model", "parts": [{"text": "I'll help you with that."}]})
            
            return content
        
        # For dictionary input, convert to proper format
        elif isinstance(content, dict):
            if "content" in content:
                main_content = content["content"]
                if format_instructions:
                    main_content = f"{main_content}\n\n{format_instructions}"
                
                if system_prompt:
                    return [
                        {"role": "user", "parts": [{"text": system_prompt}]},
                        {"role": "model", "parts": [{"text": "I'll help you with that."}]},
                        {"role": "user", "parts": [{"text": main_content}]}
                    ]
                else:
                    return main_content
            else:
                # Assume it's already properly formatted
                return content
        
        # Fallback for unsupported types
        else:
            logger.warning(f"Unsupported content type: {type(content)}. Converting to string.")
            return str(content)

    def generate_content(self, prompt, system_prompt=None, format_instructions=None, retry_on_error=True, apply_filters=True):
        """
        Enhanced general-purpose content generation with structured prompting.
        
        Args:
            prompt (str or dict): The primary content to send to the model
            system_prompt (str, optional): System instructions to guide model behavior
            format_instructions (str, optional): Instructions for output formatting
            retry_on_error (bool): Whether to retry on rate limits and transient errors
            apply_filters (bool): Whether to apply content filters to the response
            
        Returns:
            The model response
        """
        structured_prompt = self._create_structured_prompt(
            prompt, 
            system_prompt, 
            format_instructions
        )
        
        retry_attempt = 0
        max_retries = 3
        
        while True:
            try:
                response = self.model.generate_content(structured_prompt)
                self._log_request(structured_prompt, response)
                
                # Apply content filtering if requested
                if apply_filters and hasattr(response, 'text'):
                    original_text = response.text
                    filtered_text = self._filter_content(original_text)
                    
                    # If text was filtered, we need to modify the response
                    if filtered_text != original_text:
                        logger.info("Content was filtered - removed research sections or conclusion remarks")
                        # This is a bit of a hack since we can't directly modify response.text
                        # Create a custom response-like object
                        class FilteredResponse:
                            def __init__(self, original_response, new_text):
                                self.text = new_text
                                self.candidates = original_response.candidates
                                # Copy other attributes as needed
                                
                        response = FilteredResponse(response, filtered_text)
                
                return response
            except Exception as e:
                logger.error(f"Error generating content: {str(e)}")
                
                if "rate limit" in str(e).lower() and retry_on_error:
                    retry_attempt = self._handle_rate_limits(retry_attempt, max_retries)
                    continue
                elif retry_on_error and retry_attempt < max_retries:
                    retry_attempt += 1
                    wait_time = 2 ** retry_attempt
                    logger.info(f"Retrying after error ({retry_attempt}/{max_retries}), waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

    def generate_explanation(self, topic, audience="student", complexity="intermediate", format="markdown"):
        """
        Generate a detailed explanation for a given topic with customized parameters.
        
        Args:
            topic (str): The topic to explain
            audience (str): Target audience (e.g., "student", "professional", "child")
            complexity (str): Level of detail ("basic", "intermediate", "advanced")
            format (str): Output format ("text", "markdown", "bullets")
            
        Returns:
            str: The generated explanation
        """
        system_prompt = """
        You are an expert educational AI assistant. Your task is to create clear, accurate, 
        and informative explanations that are tailored to the specific audience and complexity level.
        Base your explanations on well-established facts and academic consensus.
        
        Important: Do not include sections about 'ongoing research areas' or 'areas of debate' 
        that are beyond the level being taught. Also, do not add concluding remarks like 
        'good luck with your exam' or 'remember to practice'.
        """
        
        format_map = {
            "text": "Provide explanation in clear paragraphs with appropriate transitions.",
            "markdown": "Format your response using Markdown with headings, lists, and emphasis where appropriate.",
            "bullets": "Format your response as a hierarchical bullet point list for easy scanning."
        }
        
        complexity_map = {
            "basic": "Use simple language and explain fundamental concepts only. Avoid jargon.",
            "intermediate": "Balance depth with accessibility. Define specialized terms when used.",
            "advanced": "Provide in-depth analysis including nuances, exceptions, and current research directions."
        }
        
        audience_map = {
            "student": "preparing for an exam",
            "professional": "working in this field",
            "child": "learning about this for the first time",
            "general": "with general interest in the topic"
        }
        
        prompt = f"""
        Please provide a comprehensive explanation of the following topic for a {audience} {audience_map.get(audience, "")}:
        
        Topic: {topic}
        
        Your explanation should include:
        1. Key concepts and definitions
        2. Important principles or theories
        3. Practical examples where applicable
        4. Applications and relevance
        
        {complexity_map.get(complexity, complexity_map["intermediate"])}
        {format_map.get(format, format_map["markdown"])}
        """
        
        response = self.generate_content(prompt, system_prompt=system_prompt)
        return response.text

    def generate_summary(self, text, style="concise", purpose="general", length_ratio=0.25):
        """
        Generate a customized summary of the provided text.
        
        Args:
            text (str): The text to summarize
            style (str): Summary style ("concise", "comprehensive", "bullet_points", "executive")
            purpose (str): The purpose ("general", "technical", "educational", "business")
            length_ratio (float): Target length as a ratio of original text (0.1-0.5)
            
        Returns:
            str: The generated summary
        """
        # Validate length ratio
        length_ratio = max(0.1, min(0.5, length_ratio))
        target_length = int(len(text) * length_ratio)
        
        style_map = {
            "concise": "Create a brief summary focusing only on the most critical information.",
            "comprehensive": "Create a detailed summary covering all main points and key supporting details.",
            "bullet_points": "Format the summary as bullet points highlighting key takeaways.",
            "executive": "Create an executive summary with context, key findings, and implications."
        }
        
        purpose_map = {
            "general": "for general understanding",
            "technical": "for technical audience, preserving specialized terminology",
            "educational": "for educational purposes, highlighting learning concepts",
            "business": "for business context, focusing on strategic implications"
        }
        
        system_prompt = """
        You are an expert summarization AI. Your task is to create accurate, cohesive summaries
        that capture the essential information from the original text while meeting the specified
        style and purpose requirements.
        """
        
        prompt = f"""
        Please summarize the following text {purpose_map.get(purpose, purpose_map["general"])}:
        
        {text}
        
        {style_map.get(style, style_map["concise"])}
        
        Your summary should:
        1. Highlight the main points
        2. Omit unnecessary details
        3. Maintain the core message
        4. Be approximately {int(length_ratio * 100)}% of the original length (target: ~{target_length} characters)
        5. Preserve the original meaning without adding new information
        """
        
        response = self.generate_content(prompt, system_prompt=system_prompt)
        return response.text
        
    def answer_query(self, query, context=None, citation_needed=False):
        """
        Answer a query with optional context for RAG-like functionality.
        
        Args:
            query (str): The user's question
            context (str or list, optional): Supporting context/documents
            citation_needed (bool): Whether to include citations to context
            
        Returns:
            str: The generated answer
        """
        system_prompt = """
        You are a helpful, accurate, and honest AI assistant. When answering questions:
        - If you know the answer, provide it clearly and comprehensively
        - If context is provided, use it to inform your answer
        - If citation is requested, reference relevant parts of the context
        - If you don't know or are uncertain, acknowledge the limitations
        - Do not fabricate information or sources
        """
        
        if context:
            if isinstance(context, list):
                context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])
            else:
                context_text = context
                
            prompt = f"""
            Please answer the following question using the provided context:
            
            Question: {query}
            
            Context:
            {context_text}
            
            {"Please include citations to the specific parts of the context you used in your answer." if citation_needed else ""}
            If the context doesn't contain the information needed to answer the question, please indicate that.
            """
        else:
            prompt = f"""
            Please answer the following question:
            
            Question: {query}
            
            Provide a comprehensive and accurate response. If you don't have enough information to answer confidently, 
            please acknowledge the limitations.
            """
            
        response = self.generate_content(prompt, system_prompt=system_prompt)
        return response.text