import os
from openai import OpenAI
from dotenv import load_dotenv

class LLMClient:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = os.getenv("LLM_MODEL", "qwen-max")
        self.enable_thinking = os.getenv("ENABLE_THINKING", "false").lower() in ["1", "true", "yes"]
        self.request_timeout = float(os.getenv("LLM_TIMEOUT", "180"))
        self.disabled_reason = None
        
        if not api_key:
            print("Warning: DASHSCOPE_API_KEY / OPENAI_API_KEY not found in environment variables. LLM features will be disabled.")
            self.client = None
            self.disabled_reason = "missing_api_key"
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.request_timeout,
                max_retries=1,
            )

    def generate(self, prompt, system_prompt="You are a helpful AI assistant.", enable_thinking=None, timeout=None, max_tokens=None):
        if not self.client:
            return "Error calling LLM API: LLM is disabled or API key not configured."
            
        try:
            request_kwargs = {}
            # Allow global override or per-request setting
            use_thinking = self.enable_thinking 
            if enable_thinking is not None:
                use_thinking = bool(enable_thinking)
                
            if use_thinking:
                 # Check if the provider is Aliyun/Dashscope which uses "enable_thinking"
                 # Or if it's DeepSeek/Other which might use "reasoning_effort" or similar standard
                 # For Qwen via OpenAI compat, it's often extra_body
                 if "qwen" in self.model.lower():
                     request_kwargs["extra_body"] = {"enable_thinking": True}
                 # Removed blindly adding it for all models to avoid errors if parameter is unknown
            
            if max_tokens is not None:
                request_kwargs["max_tokens"] = int(max_tokens)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                timeout=(self.request_timeout if timeout is None else float(timeout)),
                **request_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            err_msg = str(e)
            # Auto-disable on invalid key to avoid repeated 401 failures/noise.
            if "invalid_api_key" in err_msg or "Incorrect API key" in err_msg:
                self.client = None
                self.disabled_reason = "invalid_api_key"
            return f"Error calling LLM API: {e}"
