from typing import Dict, Any, Optional, TypedDict
from app.predictive_lead_score.models import LeadAnalysisResult
import json
import re

# Optional: If you're using actual protobuf objects from google-generativeai
try:
    from google.generativeai.types import GenerateContentResponse
    PROTO_AVAILABLE = True
except ImportError:
    PROTO_AVAILABLE = False




def parse_gemini_generate_content_response(response: Any) -> Optional[LeadAnalysisResult]:

    text_content = ""

    # Case 1: Real protobuf object (google-generativeai)
    if PROTO_AVAILABLE and isinstance(response, GenerateContentResponse):
        if response.candidates and response.candidates[0].content.parts:
            text_content = response.candidates[0].content.parts[0].text

    # Case 2: Dict (from .to_dict() or API simulation)
    elif isinstance(response, dict):
        candidates = response.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                text_content = parts[0]["text"]

    # Case 3: String representation (like your input)
    elif isinstance(response, str):
        # Look for ```json ... ``` block
        json_match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return {
                    "transcription": parsed.get("transcription", ""),
                    "score": parsed.get("score", 0),
                    "explanation": parsed.get("explanation", "")
                }
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract first "text" field from stringified proto
        text_match = re.search(r'"text":\s*"([^"]+)"', response.replace('\n', ' '))
        if text_match:
            text_content = text_match.group(1).replace("\\n", "\n")

    # If we have raw text content (from protobuf or dict), parse JSON inside it
    if text_content.strip():
        # Extract JSON block if wrapped in ```
        json_block_match = re.search(r"```json\s*({.*?})\s*```", text_content, re.DOTALL)
        if json_block_match:
            try:
                data = json.loads(json_block_match.group(1))
                return {
                    "transcription": data.get("transcription", ""),
                    "score": int(data.get("score", 0)),
                    "explanation": data.get("explanation", "")
                }
            except (json.JSONDecodeError, ValueError):
                return None

        # Fallback: try parsing the entire text as JSON (in case no code block)
        try:
            data = json.loads(text_content.strip())
            return {
                "transcription": data.get("transcription", ""),
                "score": int(data.get("score", 0)),
                "explanation": data.get("explanation", "")
            }
        except json.JSONDecodeError:
            pass

    return None

