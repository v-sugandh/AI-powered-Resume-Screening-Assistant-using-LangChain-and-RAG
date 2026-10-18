import time
from typing import Dict

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from models import CandidateEvaluation, ComparisonResult
from rag_pipeline import ResumeIndex

# ---------------------------------------------------------------- prompts
EVAL_TEMPLATE = """You are an expert technical recruiter screening a resume against a job description.

STRICT RULES
- Use ONLY the information in the RESUME EXCERPTS below. Do not use outside knowledge about the candidate.
- If a JD requirement is not clearly evidenced in the excerpts, list it under missing skills. Never assume or infer skills.
- Be specific: refer to actual projects, tools, roles or years found in the excerpts.

SCORING GUIDE (match_score, 0-100)
- 85-100: meets nearly all must-have requirements with strong, relevant experience
- 70-84: meets most requirements, minor gaps
- 50-69: partial fit, notable gaps
- 0-49: poor fit
Recommendation: "Strong Hire" (>=85), "Hire" (70-84), "Maybe" (50-69), "No Hire" (<50).

JOB DESCRIPTION
{job_description}

RESUME EXCERPTS (file: {candidate})
{context}

{format_instructions}
"""

COMPARE_TEMPLATE = """You are a hiring manager choosing between candidates for one role.
Base your decision ONLY on the candidate evaluations below (which were derived from their resumes).
Rank all candidates from best to worst and name the single best candidate.

JOB DESCRIPTION
{job_description}

CANDIDATE EVALUATIONS
{evaluations}

{format_instructions}
"""

QA_TEMPLATE = """Answer the recruiter's question using ONLY the resume excerpts below.
If the answer is not in the excerpts, reply exactly: "Not found in the uploaded resumes."
Mention which candidate each fact belongs to.

RESUME EXCERPTS
{context}

QUESTION: {question}

ANSWER:"""

eval_parser = PydanticOutputParser(pydantic_object=CandidateEvaluation)
compare_parser = PydanticOutputParser(pydantic_object=ComparisonResult)

EVAL_PROMPT = PromptTemplate(
    template=EVAL_TEMPLATE,
    input_variables=["job_description", "candidate", "context"],
    partial_variables={"format_instructions": eval_parser.get_format_instructions()},
)
COMPARE_PROMPT = PromptTemplate(
    template=COMPARE_TEMPLATE,
    input_variables=["job_description", "evaluations"],
    partial_variables={"format_instructions": compare_parser.get_format_instructions()},
)
QA_PROMPT = PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])


# ---------------------------------------------------------------- helpers
def _invoke_with_retry(chain, inputs: dict, retries: int = 3):
    """Free-tier APIs rate-limit and occasionally return malformed JSON, so retry."""
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (2**attempt))


# ---------------------------------------------------------------- public API
def evaluate_candidate(index: ResumeIndex, llm, candidate: str, job_description: str) -> CandidateEvaluation:
    context = index.retrieve_context(candidate, job_description)
    chain = EVAL_PROMPT | llm | eval_parser
    result: CandidateEvaluation = _invoke_with_retry(
        chain, {"job_description": job_description, "candidate": candidate, "context": context}
    )
    result.match_score = max(0, min(100, int(result.match_score)))
    result.candidate_name = candidate  # keep label consistent with the uploaded file
    return result


def compare_candidates(llm, evaluations: Dict[str, CandidateEvaluation], job_description: str) -> ComparisonResult:
    summary = "\n\n".join(
        f"### {name}\n{ev.model_dump_json(indent=2)}" for name, ev in evaluations.items()
    )
    chain = COMPARE_PROMPT | llm | compare_parser
    return _invoke_with_retry(chain, {"job_description": job_description, "evaluations": summary})


def answer_question(index: ResumeIndex, llm, question: str, candidate: str = "All candidates") -> str:
    targets = index.candidates if candidate == "All candidates" else [candidate]
    blocks = []
    for name in targets:
        docs = index.retriever(name, k=4).invoke(question)
        text = "\n".join(d.page_content for d in docs)
        blocks.append(f"[Candidate: {name}]\n{text}")
    chain = QA_PROMPT | llm | StrOutputParser()
    return _invoke_with_retry(chain, {"context": "\n\n".join(blocks), "question": question})
