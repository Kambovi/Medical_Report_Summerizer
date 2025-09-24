What:	System that read patient medical report and generates summaries for doctors.
Tech stack:
	1. RAG: Medical pdfs + FAISS vector DB
	2. Fine Tuned LLM: BioBERT or ClinicalBERT
	3. Deployment FASTAPI + AWS EC2
Why: HealthCare A is booming (GIPAA-compliant AI)