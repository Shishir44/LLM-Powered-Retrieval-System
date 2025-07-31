#!/usr/bin/env python3
"""
Enhanced Integration Testing for Phase 2 Features
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List

class EnhancedIntegrationTester:
    """Test enhanced RAG pipeline features."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_enhanced_chat(self) -> Dict[str, Any]:
        """Test enhanced chat endpoint with Phase 2 features."""
        
        test_cases = [
            {
                "name": "Simple Query",
                "message": "What is Docker?",
                "expected_features": ["query_analysis", "verification_result"]
            },
            {
                "name": "Multi-Source Query",
                "message": "Compare Docker and Kubernetes for container orchestration",
                "expected_features": ["synthesis_info", "quality_metrics"]
            },
            {
                "name": "Context-Dependent Query",
                "message": "How do I deploy it?",
                "conversation_id": "test_context_conv",
                "expected_features": ["context_info"]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"Testing: {test_case['name']}")
            
            payload = {
                "message": test_case["message"],
                "user_id": "test_user",
                "conversation_id": test_case.get("conversation_id", f"test_{int(time.time())}")
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/chat",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for expected features
                    features_present = []
                    for feature in test_case["expected_features"]:
                        if feature in data.get("metadata", {}):
                            features_present.append(feature)
                    
                    results.append({
                        "test_case": test_case["name"],
                        "status": "PASS",
                        "features_present": features_present,
                        "response_length": len(data.get("response", "")),
                        "processing_time": data.get("metadata", {}).get("processing_time", 0)
                    })
                    
                else:
                    results.append({
                        "test_case": test_case["name"],
                        "status": "FAIL",
                        "error": f"HTTP {response.status}"
                    })
            
            await asyncio.sleep(1)  # Rate limiting
        
        return {"enhanced_chat_tests": results}
    
    async def test_fact_verification(self) -> Dict[str, Any]:
        """Test standalone fact verification endpoint."""
        
        test_cases = [
            {
                "name": "Factual Statement",
                "text": "Docker was first released in 2013 by Docker Inc.",
                "expected_risk": "low"
            },
            {
                "name": "Potentially Inaccurate Statement",
                "text": "Docker was invented in 1995 and runs on Windows only.",
                "expected_risk": "high"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            payload = {
                "response_text": test_case["text"],
                "verification_level": "quick"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/verify",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    verification = data.get("verification_result", {})
                    
                    results.append({
                        "test_case": test_case["name"],
                        "status": "PASS",
                        "risk_level": verification.get("hallucination_analysis", {}).get("hallucination_risk", "unknown"),
                        "confidence": verification.get("overall_confidence", 0)
                    })
                else:
                    results.append({
                        "test_case": test_case["name"],
                        "status": "FAIL",
                        "error": f"HTTP {response.status}"
                    })
        
        return {"fact_verification_tests": results}
    
    async def test_multi_source_synthesis(self) -> Dict[str, Any]:
        """Test multi-source synthesis endpoint."""
        
        test_sources = [
            {
                "id": "source1",
                "content": "Docker is a containerization platform that helps developers package applications.",
                "title": "Docker Overview"
            },
            {
                "id": "source2", 
                "content": "Docker containers are lightweight and portable, making deployment easier.",
                "title": "Docker Benefits"
            }
        ]
        
        payload = {
            "sources": test_sources,
            "query": "What is Docker and what are its benefits?",
            "synthesis_strategy": "convergent"
        }
        
        async with self.session.post(
            f"{self.base_url}/api/v1/synthesize",
            json=payload
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                synthesis = data.get("synthesis_result", {})
                
                return {
                    "multi_source_synthesis_test": {
                        "status": "PASS",
                        "strategy_used": synthesis.get("strategy", "unknown"),
                        "source_count": len(test_sources),
                        "synthesis_length": len(synthesis.get("synthesized_response", ""))
                    }
                }
            else:
                return {
                    "multi_source_synthesis_test": {
                        "status": "FAIL",
                        "error": f"HTTP {response.status}"
                    }
                }
    
    async def test_context_management(self) -> Dict[str, Any]:
        """Test enhanced context management."""
        
        conversation_id = f"context_test_{int(time.time())}"
        
        # Send multiple messages to build context
        messages = [
            "What is Docker?",
            "How do I install it?",
            "What are the best practices?",
            "Can you explain the previous answer in more detail?"
        ]
        
        results = []
        
        for i, message in enumerate(messages):
            payload = {
                "message": message,
                "user_id": "context_test_user",
                "conversation_id": conversation_id
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/chat",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    context_info = data.get("metadata", {}).get("context_info", {})
                    
                    results.append({
                        "message_number": i + 1,
                        "context_size": context_info.get("context_size", 0),
                        "context_strategy": context_info.get("context_strategy", "unknown")
                    })
        
        return {"context_management_tests": results}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all enhanced integration tests."""
        
        print("🧪 Running Enhanced Integration Tests...")
        
        all_results = {}
        
        # Test enhanced chat
        print("Testing enhanced chat endpoint...")
        all_results.update(await self.test_enhanced_chat())
        
        # Test fact verification
        print("Testing fact verification...")
        all_results.update(await self.test_fact_verification())
        
        # Test multi-source synthesis
        print("Testing multi-source synthesis...")
        all_results.update(await self.test_multi_source_synthesis())
        
        # Test context management
        print("Testing context management...")
        all_results.update(await self.test_context_management())
        
        return all_results

async def main():
    """Run enhanced integration tests."""
    
    async with EnhancedIntegrationTester() as tester:
        results = await tester.run_all_tests()
        
        print("\n" + "="*50)
        print("🎯 ENHANCED INTEGRATION TEST RESULTS")
        print("="*50)
        
        for test_category, test_results in results.items():
            print(f"\n📋 {test_category.replace('_', ' ').title()}:")
            
            if isinstance(test_results, list):
                for result in test_results:
                    status_emoji = "✅" if result.get("status") == "PASS" else "❌"
                    print(f"  {status_emoji} {result.get('test_case', 'Unknown')}")
            else:
                status_emoji = "✅" if test_results.get("status") == "PASS" else "❌"
                print(f"  {status_emoji} {test_results}")
        
        print(f"\n🎉 Enhanced integration testing complete!")

if __name__ == "__main__":
    asyncio.run(main())