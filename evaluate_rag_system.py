#!/usr/bin/env python3
"""
RAG System Evaluation Framework
This script evaluates the RAG system against the requirements from rag_system_requirements.md
"""

import json
import requests
import time
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio

@dataclass
class TestQuery:
    """Test query with expected characteristics."""
    query: str
    query_type: str  # simple_factual, contextual, multi_hop, complex_reasoning
    expected_topics: List[str]
    difficulty: str  # easy, medium, hard
    expected_sources: int  # minimum expected sources

# Test queries from the requirements document
TEST_QUERIES = [
    # Simple Fact-Based Queries
    TestQuery(
        query="What is Docker?",
        query_type="simple_factual",
        expected_topics=["docker", "containerization", "devops"],
        difficulty="easy",
        expected_sources=1
    ),
    TestQuery(
        query="Define zero trust security.",
        query_type="simple_factual", 
        expected_topics=["zero trust", "security", "authentication"],
        difficulty="easy",
        expected_sources=1
    ),
    TestQuery(
        query="What is the purpose of an API?",
        query_type="simple_factual",
        expected_topics=["api", "integration", "communication"],
        difficulty="easy", 
        expected_sources=1
    ),
    
    # Moderate Contextual Queries
    TestQuery(
        query="How does Docker help in CI/CD pipelines?",
        query_type="contextual",
        expected_topics=["docker", "ci/cd", "deployment", "automation"],
        difficulty="medium",
        expected_sources=2
    ),
    TestQuery(
        query="Why is green computing important?",
        query_type="contextual",
        expected_topics=["green computing", "sustainability", "energy"],
        difficulty="medium",
        expected_sources=1
    ),
    TestQuery(
        query="What are neural networks used for?",
        query_type="contextual",
        expected_topics=["neural networks", "ai", "applications"],
        difficulty="medium",
        expected_sources=1
    ),
    
    # Multi-Hop or Comparative Queries
    TestQuery(
        query="Compare traditional APIs with Web3 smart contracts.",
        query_type="multi_hop",
        expected_topics=["api", "web3", "smart contracts", "comparison"],
        difficulty="hard",
        expected_sources=2
    ),
    TestQuery(
        query="How does zero trust differ from VPNs?",
        query_type="multi_hop", 
        expected_topics=["zero trust", "vpn", "security", "comparison"],
        difficulty="hard",
        expected_sources=1
    ),
    
    # Complex Reasoning Queries
    TestQuery(
        query="How would an e-commerce platform benefit from CI/CD and Docker?",
        query_type="complex_reasoning",
        expected_topics=["ci/cd", "docker", "e-commerce", "benefits"],
        difficulty="hard",
        expected_sources=2
    ),
    TestQuery(
        query="Design a system using serverless, 5G, and AI for agriculture monitoring.",
        query_type="complex_reasoning",
        expected_topics=["serverless", "5g", "ai", "agriculture", "system design"],
        difficulty="hard", 
        expected_sources=3
    )
]

@dataclass
class EvaluationResult:
    """Results of evaluating a single query."""
    query: str
    query_type: str
    response: str
    sources: List[Dict[str, Any]]
    processing_time: float
    success: bool
    error_message: str = ""
    
    # Evaluation metrics
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    source_quality_score: float = 0.0
    citation_score: float = 0.0
    overall_score: float = 0.0

class RAGEvaluator:
    """Comprehensive RAG system evaluator."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.knowledge_base_url = f"{base_url}/knowledge"
        self.conversation_url = f"{base_url}/conversation"
        
    def check_system_health(self) -> Dict[str, bool]:
        """Check if all services are running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json().get("services", {})
            return {}
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {}
    
    def load_sample_documents(self) -> bool:
        """Ensure sample documents are loaded."""
        try:
            # Check if documents exist
            stats_response = requests.get(f"{self.knowledge_base_url}/stats", timeout=10)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                if stats.get("total_documents", 0) > 0:
                    print(f"✅ Found {stats['total_documents']} documents in knowledge base")
                    return True
            
            print("📖 Loading sample documents...")
            # Load documents from sample file
            try:
                with open("sample_documents.json", "r") as f:
                    documents = json.load(f)
                
                response = requests.post(
                    f"{self.knowledge_base_url}/documents/bulk",
                    json=documents,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Loaded {result['created_count']} documents successfully")
                    return True
                else:
                    print(f"❌ Failed to load documents: {response.status_code}")
                    return False
                    
            except FileNotFoundError:
                print("❌ sample_documents.json not found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return False
    
    def query_rag_system(self, query: str) -> Tuple[str, List[Dict], float, bool, str]:
        """Query the RAG system and return results."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.conversation_url}/api/v1/chat",
                json={
                    "message": query,
                    "conversation_id": None,
                    "context": {}
                },
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return (
                    data.get("response", ""),
                    data.get("sources", []),
                    processing_time,
                    True,
                    ""
                )
            else:
                return (
                    "",
                    [],
                    processing_time,
                    False,
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            return ("", [], processing_time, False, str(e))
    
    def evaluate_response_quality(self, test_query: TestQuery, response: str, sources: List[Dict]) -> Dict[str, float]:
        """Evaluate the quality of a RAG response."""
        scores = {}
        
        # 1. Relevance Score (0-1)
        relevance_score = 0.0
        query_lower = test_query.query.lower()
        response_lower = response.lower()
        
        # Check if key topics are mentioned
        topic_matches = sum(1 for topic in test_query.expected_topics if topic.lower() in response_lower)
        relevance_score = min(topic_matches / len(test_query.expected_topics), 1.0)
        scores["relevance_score"] = relevance_score
        
        # 2. Completeness Score (0-1)
        completeness_score = 0.0
        if len(response) > 50:  # Basic length check
            completeness_score += 0.3
        if any(topic in response_lower for topic in test_query.expected_topics):
            completeness_score += 0.4
        if "." in response and len(response.split(".")) >= 2:  # Multiple sentences
            completeness_score += 0.3
        scores["completeness_score"] = min(completeness_score, 1.0)
        
        # 3. Source Quality Score (0-1)
        source_quality_score = 0.0
        if len(sources) >= test_query.expected_sources:
            source_quality_score += 0.5
        if sources and any(source.get("score", 0) > 0.7 for source in sources):
            source_quality_score += 0.3
        if sources and any(any(tag in test_query.expected_topics for tag in source.get("metadata", {}).get("tags", [])) for source in sources):
            source_quality_score += 0.2
        scores["source_quality_score"] = source_quality_score
        
        # 4. Citation Score (0-1) 
        citation_score = 0.0
        if "[source:" in response_lower or "source:" in response_lower:
            citation_score += 0.5
        if "sources:" in response_lower:
            citation_score += 0.3
        if sources and len(sources) > 0:
            citation_score += 0.2
        scores["citation_score"] = min(citation_score, 1.0)
        
        # 5. Accuracy Score (manual approximation based on content relevance)
        accuracy_score = relevance_score * 0.8 + (1.0 if len(response) > 20 else 0.0) * 0.2
        scores["accuracy_score"] = accuracy_score
        
        # Overall score
        scores["overall_score"] = (
            relevance_score * 0.3 +
            completeness_score * 0.25 +
            source_quality_score * 0.2 +
            citation_score * 0.15 +
            accuracy_score * 0.1
        )
        
        return scores
    
    def run_evaluation(self) -> List[EvaluationResult]:
        """Run complete evaluation on all test queries."""
        print("🚀 Starting RAG System Evaluation")
        print("=" * 60)
        
        # Check system health
        health = self.check_system_health()
        if not all(health.values()):
            print("❌ Some services are not healthy. Please check the system.")
            for service, healthy in health.items():
                status = "✅" if healthy else "❌"
                print(f"  {status} {service}")
            return []
        
        # Load documents
        if not self.load_sample_documents():
            print("❌ Failed to load sample documents")
            return []
        
        # Run tests
        results = []
        total_queries = len(TEST_QUERIES)
        
        print(f"\n📝 Running {total_queries} test queries...")
        print("-" * 60)
        
        for i, test_query in enumerate(TEST_QUERIES):
            print(f"\n🔍 Query {i+1}/{total_queries}: {test_query.query}")
            print(f"Type: {test_query.query_type} | Difficulty: {test_query.difficulty}")
            
            # Query the system
            response, sources, processing_time, success, error = self.query_rag_system(test_query.query)
            
            if not success:
                print(f"❌ Query failed: {error}")
                results.append(EvaluationResult(
                    query=test_query.query,
                    query_type=test_query.query_type,
                    response="",
                    sources=[],
                    processing_time=processing_time,
                    success=False,
                    error_message=error
                ))
                continue
            
            # Evaluate response quality
            quality_scores = self.evaluate_response_quality(test_query, response, sources)
            
            result = EvaluationResult(
                query=test_query.query,
                query_type=test_query.query_type,
                response=response,
                sources=sources,
                processing_time=processing_time,
                success=True,
                **quality_scores
            )
            
            results.append(result)
            
            # Print results
            print(f"✅ Success | Time: {processing_time:.2f}s | Overall Score: {quality_scores['overall_score']:.2f}")
            print(f"📊 Relevance: {quality_scores['relevance_score']:.2f} | Completeness: {quality_scores['completeness_score']:.2f} | Citations: {quality_scores['citation_score']:.2f}")
            print(f"📚 Sources: {len(sources)} | Expected: {test_query.expected_sources}")
            
            if len(response) > 100:
                print(f"💬 Response: {response[:100]}...")
            else:
                print(f"💬 Response: {response}")
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not results:
            return {"error": "No evaluation results available"}
        
        # Filter successful results for calculations
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful queries"}
        
        # Overall statistics
        total_queries = len(results)
        success_rate = len(successful_results) / total_queries
        
        avg_processing_time = sum(r.processing_time for r in successful_results) / len(successful_results)
        avg_overall_score = sum(r.overall_score for r in successful_results) / len(successful_results)
        
        # Scores by category
        score_breakdown = {
            "relevance": sum(r.relevance_score for r in successful_results) / len(successful_results),
            "completeness": sum(r.completeness_score for r in successful_results) / len(successful_results),
            "accuracy": sum(r.accuracy_score for r in successful_results) / len(successful_results),
            "source_quality": sum(r.source_quality_score for r in successful_results) / len(successful_results),
            "citation": sum(r.citation_score for r in successful_results) / len(successful_results)
        }
        
        # Performance by query type
        query_type_performance = {}
        for result in successful_results:
            if result.query_type not in query_type_performance:
                query_type_performance[result.query_type] = []
            query_type_performance[result.query_type].append(result.overall_score)
        
        # Average by query type
        for query_type, scores in query_type_performance.items():
            query_type_performance[query_type] = sum(scores) / len(scores)
        
        # Failed queries
        failed_queries = [
            {"query": r.query, "error": r.error_message}
            for r in results if not r.success
        ]
        
        return {
            "evaluation_summary": {
                "total_queries": total_queries,
                "successful_queries": len(successful_results),
                "success_rate": success_rate,
                "average_processing_time": avg_processing_time,
                "average_overall_score": avg_overall_score
            },
            "score_breakdown": score_breakdown,
            "query_type_performance": query_type_performance,
            "failed_queries": failed_queries,
            "detailed_results": [
                {
                    "query": r.query,
                    "query_type": r.query_type,
                    "success": r.success,
                    "processing_time": r.processing_time,
                    "overall_score": r.overall_score,
                    "scores": {
                        "relevance": r.relevance_score,
                        "completeness": r.completeness_score,
                        "accuracy": r.accuracy_score,
                        "source_quality": r.source_quality_score,
                        "citation": r.citation_score
                    },
                    "sources_count": len(r.sources),
                    "response_length": len(r.response)
                }
                for r in results
            ]
        }

def main():
    print("🧪 RAG System Evaluation Framework")
    print("Based on requirements from rag_system_requirements.md")
    print("=" * 60)
    
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if not results:
        print("❌ Evaluation failed - no results generated")
        sys.exit(1)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 EVALUATION REPORT")
    print("=" * 60)
    
    report = evaluator.generate_report(results)
    
    if "error" in report:
        print(f"❌ {report['error']}")
        sys.exit(1)
    
    # Print summary
    summary = report["evaluation_summary"]
    print(f"\n📈 Overall Performance:")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    print(f"  Average Score: {summary['average_overall_score']:.2f}/1.0")
    print(f"  Average Processing Time: {summary['average_processing_time']:.2f}s")
    print(f"  Successful Queries: {summary['successful_queries']}/{summary['total_queries']}")
    
    # Print score breakdown
    print(f"\n📊 Score Breakdown:")
    breakdown = report["score_breakdown"]
    for metric, score in breakdown.items():
        print(f"  {metric.title()}: {score:.2f}")
    
    # Print performance by query type
    print(f"\n🎯 Performance by Query Type:")
    for query_type, avg_score in report["query_type_performance"].items():
        print(f"  {query_type.replace('_', ' ').title()}: {avg_score:.2f}")
    
    # Print failed queries if any
    if report["failed_queries"]:
        print(f"\n❌ Failed Queries ({len(report['failed_queries'])}):")
        for failed in report["failed_queries"]:
            print(f"  - {failed['query']}: {failed['error']}")
    
    # Save detailed report
    with open("rag_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: rag_evaluation_report.json")
    
    # Final assessment
    overall_score = summary['average_overall_score']
    if overall_score >= 0.8:
        print(f"\n🎉 Excellent! RAG system is performing very well (Score: {overall_score:.2f})")
    elif overall_score >= 0.6:
        print(f"\n✅ Good! RAG system is performing well (Score: {overall_score:.2f})")
    elif overall_score >= 0.4:
        print(f"\n⚠️  Fair! RAG system needs improvement (Score: {overall_score:.2f})")
    else:
        print(f"\n🔧 Poor! RAG system requires significant improvements (Score: {overall_score:.2f})")
    
    print("\nRecommendations based on requirements:")
    if breakdown["relevance"] < 0.7:
        print("- Improve semantic search and query understanding")
    if breakdown["completeness"] < 0.7:
        print("- Enhance response generation for more complete answers")
    if breakdown["citation"] < 0.7:
        print("- Improve source citation and grounding")
    if breakdown["source_quality"] < 0.7:
        print("- Enhance document retrieval and ranking")

if __name__ == "__main__":
    main()