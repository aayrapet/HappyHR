import os
import sys
from dotenv import load_dotenv

# Load the environment variables from .env
load_dotenv()

def test_google_cloud_auth():
    print("--- Test Google Cloud Auth ---")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    print(f"GOOGLE_CLOUD_PROJECT: {project_id}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")
    
    if not project_id or not creds_path:
        print("\n❌ Missing GOOGLE_CLOUD_PROJECT or GOOGLE_APPLICATION_CREDENTIALS in .env")
        return
        
    if not os.path.exists(creds_path):
        print(f"\n❌ Credential file not found at: {creds_path}")
        return
        
    print("\n✅ Environment variables are set and file exists. Testing API call...")
    
    try:
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        
        # Initialize client
        client = SpeechClient()
        
        # We will try to make a dummy recognize request to see if we get a 403 Permission Denied
        # Create a tiny 1-second silent WAV file in memory
        import wave
        import io
        import struct
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b'\x00' * 32000) # 1 second of silence
            
        audio_content = wav_io.getvalue()
        
        request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/global/recognizers/_",
            config=cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    audio_channel_count=1,
                ),
                language_codes=["en-US"],
                model="latest_short",
            ),
            content=audio_content,
        )
        
        print("Sending test request to Google Cloud Speech-to-Text V2...")
        response = client.recognize(request=request)
        print("\n✅ SUCCESS: The API call succeeded! Your service account has the correct permissions.")
        
    except Exception as e:
        print(f"\n❌ FAILED: API call returned an error.")
        print(f"Error Details: {e}")
        print("\nRemedy: Go to Google Cloud Console > IAM & Admin > IAM.")
        print("Click 'GRANT ACCESS' (Add). Add your service account email and give it the 'Cloud Speech Client' OR 'Editor' role.")

if __name__ == "__main__":
    test_google_cloud_auth()
