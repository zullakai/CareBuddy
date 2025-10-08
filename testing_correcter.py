import google.generativeai as genai

genai.configure(api_key="AIzaSyAccIZJvoYZ7Ctrkjo7psGhLv-g1shJaus")

model = genai.GenerativeModel(
    "models/gemini-2.5-flash",
    system_instruction="You are a grammar correction assistant. Return only the corrected sentence."
)

text = "I GO SCHOOL"
response = model.generate_content(text)
print("Detected:", text)
print("Corrected:", response.text)
