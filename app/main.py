from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

from app.services.file_processor import FileProcessor
from app.services.cv_analyzer import CVAnalyzer
from app.models import AnalysisResult
from app.services.linkedin_fetcher import LinkedInFetcher

# Load environment variables
load_dotenv()

app = FastAPI(title="CV Matcher API", version="1.1.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
file_processor = FileProcessor()
cv_analyzer = CVAnalyzer()
linkedin_fetcher = LinkedInFetcher()

@app.get("/")
async def root():
    return {"message": "CV Matcher API is running"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_cv(
    job_description: str = Form(...),
    cv_file: UploadFile | None = File(None),
    linkedin_url: str | None = Form(None),
):
    """
    Analyze CV and/or LinkedIn profile against job description and return matching results
    """
    try:
        if (cv_file is None or not getattr(cv_file, "filename", "")) and (not linkedin_url or not linkedin_url.strip()):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CV or provide a LinkedIn URL."
            )

        sources_text: list[str] = []

        # Extract text from uploaded CV if provided
        if cv_file and getattr(cv_file, "filename", ""):
            if not cv_file.filename.lower().endswith((".pdf", ".doc", ".docx")):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF, DOC, and DOCX files are supported"
                )
            cv_text = await file_processor.extract_text(cv_file)
            if cv_text and cv_text.strip():
                sources_text.append(f"CV Document:\n{cv_text}")

        # Fetch LinkedIn profile text if URL provided
        if linkedin_url and linkedin_url.strip():
            li_text = await linkedin_fetcher.fetch_profile_text(linkedin_url.strip())
            if li_text and li_text.strip():
                sources_text.append(f"LinkedIn Profile:\n{li_text}")

        if not sources_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the provided inputs."
            )

        combined_text = "\n\n---\n\n".join(sources_text)

        # Analyze combined content against job description
        analysis_result = await cv_analyzer.analyze(combined_text, job_description)

        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cv-matcher-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 