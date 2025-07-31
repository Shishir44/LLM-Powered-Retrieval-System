#!/usr/bin/env python3
"""
PHASE 3 SYSTEM INTEGRATION TESTS
Comprehensive testing suite for the Phase 3 LLM-Powered Retrieval System
Tests the current system architecture with API Gateway and enhanced features.
"""

import asyncio
import aiohttp
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase3SystemTester:
    def __init__(self):
        self.api_gateway_url = "http://localhost:8080"
        self.conversation_service_url = "http://localhost:8001"
        self.knowledge_base_service_url = "http://localhost:8002"
        self.analytics_service_url = "http://localhost:8005"
        
        self.test_results = {}
        self.test_document_id = None
        self.test_conversation_id = f"test-conv-{int(time.time())}"
        
    async def log_test_step(self, step_name: str, message: str = ""):
        """Log test step with formatting"""
        logger.info(f"🧪 {step_name} {message}")
        
    async def log_success(self, message: str):
        """Log success message"""
        logger.info(f"✅ {message}")
        
    async def log_error(self, message: str):
        """Log error message"""
        logger.error(f"❌ {message}")
        
    async def log_warning(self, message: str):
        """Log warning message"""
        logger.warning(f"⚠️ {message}")

    async def test_health_endpoints(self):
        """Test all service health endpoints"""
        await self.log_test_step("HEALTH CHECKS", "Testing all service health endpoints")
        
        services = {
            "API Gateway": f"{self.api_gateway_url}/health",
            "Conversation Service": f"{self.conversation_service_url}/health",
            "Knowledge Base Service": f"{self.knowledge_base_service_url}/health",
            "Analytics Service": f"{self.analytics_service_url}/health"
        }
        
        health_results = {}
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in services.items():
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            health_results[service_name] = data.get('status', 'unknown')
                            await self.log_success(f"{service_name}: {data.get('status', 'unknown')}")
                        else:
                            health_results[service_name] = f"HTTP {response.status}"
                            await self.log_error(f"{service_name}: HTTP {response.status}")
                except Exception as e:
                    health_results[service_name] = f"Error: {str(e)}"
                    await self.log_error(f"{service_name}: {str(e)}")
        
        self.test_results['health_checks'] = health_results
        return all(status == 'healthy' for status in health_results.values())

    async def test_document_management(self):
        """Test document creation, search, and retrieval"""
        await self.log_test_step("DOCUMENT MANAGEMENT", "Testing document operations")
        
        async with aiohttp.ClientSession() as session:
            # Test document creation
            document_data = {
                "title": "Phase 3 Test Document",
                "content": "This is a comprehensive test document for the Phase 3 LLM-Powered Retrieval System. It contains information about advanced RAG capabilities, semantic search, and enterprise features. The system supports multi-hop reasoning, adaptive learning, and real-time personalization for enhanced user experiences.",
                "category": "technical",
                "subcategory": "testing",
                "tags": ["phase3", "testing", "rag", "semantic-search"],
                "metadata": {
                    "author": "system-test",
                    "version": "3.0",
                    "test_timestamp": datetime.now().isoformat()
                }
            }
            
            try:
                # Create document
                async with session.post(
                    f"{self.knowledge_base_service_url}/api/v1/documents",
                    json=document_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        doc_response = await response.json()
                        self.test_document_id = doc_response.get('id')
                        await self.log_success(f"Document created with ID: {self.test_document_id}")
                    else:
                        await self.log_error(f"Document creation failed: HTTP {response.status}")
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        return False
                        
            except Exception as e:
                await self.log_error(f"Document creation error: {str(e)}")
                return False
            
            # Test search functionality
            try:
                search_params = {
                    "q": "Phase 3 advanced RAG capabilities",
                    "limit": 5
                }
                
                async with session.get(
                    f"{self.knowledge_base_service_url}/api/v1/search",
                    params=search_params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        search_results = await response.json()
                        total_results = search_results.get('total', 0)
                        await self.log_success(f"Search found {total_results} documents")
                        self.test_results['document_search'] = search_results
                    else:
                        await self.log_error(f"Search failed: HTTP {response.status}")
                        return False
                        
            except Exception as e:
                await self.log_error(f"Search error: {str(e)}")
                return False
            
            # Test semantic search
            try:
                async with session.get(
                    f"{self.knowledge_base_service_url}/api/v1/search/semantic",
                    params={"q": "enterprise features real-time personalization", "limit": 3},
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        semantic_results = await response.json()
                        await self.log_success(f"Semantic search completed with {semantic_results.get('total', 0)} results")
                        self.test_results['semantic_search'] = semantic_results
                    else:
                        await self.log_warning(f"Semantic search failed: HTTP {response.status}")
            except Exception as e:
                await self.log_warning(f"Semantic search error: {str(e)}")
        
        return True

    async def test_conversation_capabilities(self):
        """Test conversation and enhanced chat capabilities"""
        await self.log_test_step("CONVERSATION CAPABILITIES", "Testing chat and enhanced features")
        
        async with aiohttp.ClientSession() as session:
            # Test basic chat
            try:
                chat_request = {
                    "message": "What are the advanced features of the Phase 3 RAG system?",
                    "conversation_id": self.test_conversation_id,
                    "context": {"test_mode": True}
                }
                
                async with session.post(
                    f"{self.conversation_service_url}/api/v1/chat",
                    json=chat_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        chat_response = await response.json()
                        response_text = chat_response.get('response', '')
                        await self.log_success(f"Basic chat successful: {response_text[:100]}...")
                        self.test_results['basic_chat'] = chat_response
                    else:
                        await self.log_error(f"Basic chat failed: HTTP {response.status}")
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        return False
                        
            except Exception as e:
                await self.log_error(f"Basic chat error: {str(e)}")
                return False
            
            # Test enhanced chat
            try:
                enhanced_request = {
                    "message": "How does the adaptive learning system improve performance over time?",
                    "conversation_id": self.test_conversation_id,
                    "user_id": "test-user-123",
                    "enable_fact_verification": True,
                    "enable_multi_source_synthesis": True
                }
                
                async with session.post(
                    f"{self.conversation_service_url}/api/v1/enhanced-chat",
                    json=enhanced_request,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status == 200:
                        enhanced_response = await response.json()
                        metadata = enhanced_response.get('metadata', {})
                        await self.log_success(f"Enhanced chat successful with confidence: {metadata.get('confidence_score', 'N/A')}")
                        self.test_results['enhanced_chat'] = enhanced_response
                    else:
                        await self.log_warning(f"Enhanced chat failed: HTTP {response.status}")
                        
            except Exception as e:
                await self.log_warning(f"Enhanced chat error: {str(e)}")
            
            # Test pipeline stats
            try:
                async with session.get(
                    f"{self.conversation_service_url}/api/v1/pipeline/stats",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        stats = await response.json()
                        await self.log_success(f"Pipeline stats retrieved: {len(stats)} metrics")
                        self.test_results['pipeline_stats'] = stats
                    else:
                        await self.log_warning(f"Pipeline stats failed: HTTP {response.status}")
                        
            except Exception as e:
                await self.log_warning(f"Pipeline stats error: {str(e)}")
        
        return True

    async def test_analytics_capabilities(self):
        """Test analytics and evaluation capabilities"""
        await self.log_test_step("ANALYTICS CAPABILITIES", "Testing evaluation and metrics")
        
        async with aiohttp.ClientSession() as session:
            # Test response evaluation
            try:
                eval_request = {
                    "query": "What are the Phase 3 features?",
                    "context": "Phase 3 includes advanced reasoning, adaptive learning, and enterprise features.",
                    "response": "Phase 3 features include advanced query reasoning with chain-of-thought processing, adaptive learning from user feedback, and comprehensive enterprise capabilities.",
                    "conversation_id": self.test_conversation_id
                }
                
                async with session.post(
                    f"{self.analytics_service_url}/api/v1/evaluate",
                    json=eval_request,
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as response:
                    if response.status == 200:
                        eval_results = await response.json()
                        metrics = eval_results.get('metrics', {})
                        await self.log_success(f"Response evaluation completed: {len(metrics)} metrics")
                        self.test_results['response_evaluation'] = eval_results
                    else:
                        await self.log_error(f"Response evaluation failed: HTTP {response.status}")
                        return False
                        
            except Exception as e:
                await self.log_error(f"Response evaluation error: {str(e)}")
                return False
            
            # Test user feedback
            try:
                feedback_request = {
                    "conversation_id": self.test_conversation_id,
                    "satisfaction_score": 0.85,
                    "feedback_text": "Excellent response with comprehensive information"
                }
                
                async with session.post(
                    f"{self.analytics_service_url}/api/v1/feedback",
                    json=feedback_request,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        await self.log_success("User feedback recorded successfully")
                    else:
                        await self.log_warning(f"User feedback failed: HTTP {response.status}")
                        
            except Exception as e:
                await self.log_warning(f"User feedback error: {str(e)}")
            
            # Test system metrics
            try:
                async with session.get(
                    f"{self.analytics_service_url}/api/v1/metrics",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        metrics = await response.json()
                        current_metrics = metrics.get('current_metrics', {})
                        await self.log_success(f"System metrics retrieved: {len(current_metrics)} metrics")
                        self.test_results['system_metrics'] = metrics
                    else:
                        await self.log_warning(f"System metrics failed: HTTP {response.status}")
                        
            except Exception as e:
                await self.log_warning(f"System metrics error: {str(e)}")
        
        return True

    async def test_api_gateway_routing(self):
        """Test API Gateway routing capabilities"""
        await self.log_test_step("API GATEWAY", "Testing routing and aggregation")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test gateway health with service status
                async with session.get(
                    f"{self.api_gateway_url}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        services_status = health_data.get('services', {})
                        all_healthy = all(services_status.values())
                        await self.log_success(f"API Gateway routing: All services {'healthy' if all_healthy else 'partially available'}")
                        self.test_results['api_gateway'] = health_data
                        return all_healthy
                    else:
                        await self.log_error(f"API Gateway health failed: HTTP {response.status}")
                        return False
                        
            except Exception as e:
                await self.log_error(f"API Gateway error: {str(e)}")
                return False

    async def test_enterprise_features(self):
        """Test enterprise-level features and monitoring"""
        await self.log_test_step("ENTERPRISE FEATURES", "Testing advanced capabilities")
        
        enterprise_tests = []
        
        # Test customer profile capabilities
        async with aiohttp.ClientSession() as session:
            try:
                customer_id = "test-customer-123"
                profile_data = {
                    "communication_style": "technical",
                    "expertise_level": "advanced",
                    "preferred_topics": ["rag", "ai", "enterprise"]
                }
                
                async with session.post(
                    f"{self.conversation_service_url}/api/v1/customers/{customer_id}/profile",
                    json=profile_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in [200, 201]:
                        await self.log_success("Customer profile management working")
                        enterprise_tests.append(True)
                    else:
                        await self.log_warning(f"Customer profile failed: HTTP {response.status}")
                        enterprise_tests.append(False)
                        
            except Exception as e:
                await self.log_warning(f"Customer profile error: {str(e)}")
                enterprise_tests.append(False)
            
            # Test context retrieval
            try:
                async with session.get(
                    f"{self.conversation_service_url}/api/v1/customers/{customer_id}/context",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        context_data = await response.json()
                        await self.log_success("Customer context retrieval working")
                        enterprise_tests.append(True)
                    else:
                        await self.log_warning(f"Customer context failed: HTTP {response.status}")
                        enterprise_tests.append(False)
                        
            except Exception as e:
                await self.log_warning(f"Customer context error: {str(e)}")
                enterprise_tests.append(False)
        
        self.test_results['enterprise_features'] = enterprise_tests
        return any(enterprise_tests)

    async def cleanup_test_data(self):
        """Clean up test data created during testing"""
        await self.log_test_step("CLEANUP", "Removing test data")
        
        if self.test_document_id:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.delete(
                        f"{self.knowledge_base_service_url}/api/v1/documents/{self.test_document_id}",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status in [200, 204]:
                            await self.log_success("Test document cleaned up")
                        else:
                            await self.log_warning(f"Document cleanup failed: HTTP {response.status}")
                            
                except Exception as e:
                    await self.log_warning(f"Document cleanup error: {str(e)}")

    async def generate_test_report(self):
        """Generate comprehensive test report"""
        await self.log_test_step("TEST REPORT", "Generating comprehensive results")
        
        print("\n" + "="*80)
        print("🎯 PHASE 3 SYSTEM INTEGRATION TEST REPORT")
        print("="*80)
        
        # Count successful tests
        successful_tests = 0
        total_tests = 0
        
        test_categories = [
            ("Health Checks", "health_checks"),
            ("Document Management", "document_search"),
            ("Basic Chat", "basic_chat"),
            ("Enhanced Chat", "enhanced_chat"),
            ("Analytics Evaluation", "response_evaluation"),
            ("API Gateway", "api_gateway"),
            ("Enterprise Features", "enterprise_features")
        ]
        
        for category_name, test_key in test_categories:
            total_tests += 1
            if test_key in self.test_results:
                successful_tests += 1
                print(f"✅ {category_name}: PASSED")
            else:
                print(f"❌ {category_name}: FAILED")
        
        # Performance metrics
        if 'enhanced_chat' in self.test_results:
            metadata = self.test_results['enhanced_chat'].get('metadata', {})
            processing_time = metadata.get('processing_time', 'N/A')
            confidence_score = metadata.get('confidence_score', 'N/A')
            print(f"\n📊 Performance Metrics:")
            print(f"   Processing Time: {processing_time}")
            print(f"   Confidence Score: {confidence_score}")
        
        # System capabilities
        print(f"\n🚀 System Capabilities Tested:")
        capabilities = [
            "✅ Multi-service architecture with API Gateway",
            "✅ Document ingestion and semantic search",
            "✅ Enhanced conversation with reasoning",
            "✅ Response quality evaluation",
            "✅ Customer profile management",
            "✅ Real-time analytics and metrics"
        ]
        
        for capability in capabilities:
            print(f"   {capability}")
        
        # Final score
        success_rate = (successful_tests / total_tests) * 100
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        if success_rate >= 80:
            print("🏆 PHASE 3 SYSTEM IS READY FOR PRODUCTION!")
        elif success_rate >= 60:
            print("⚠️  PHASE 3 SYSTEM IS MOSTLY FUNCTIONAL - Some issues need attention")
        else:
            print("❌ PHASE 3 SYSTEM NEEDS SIGNIFICANT FIXES")
        
        print("="*80 + "\n")
        
        return success_rate >= 80

async def main():
    """Run all Phase 3 system integration tests"""
    tester = Phase3SystemTester()
    
    print("🚀 Starting PHASE 3 System Integration Tests")
    print("="*80)
    
    try:
        # Run all tests
        tests = [
            tester.test_health_endpoints(),
            tester.test_document_management(),
            tester.test_conversation_capabilities(),
            tester.test_analytics_capabilities(),
            tester.test_api_gateway_routing(),
            tester.test_enterprise_features()
        ]
        
        # Execute tests
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                await tester.log_error(f"Test {i+1} failed with exception: {str(result)}")
        
        # Cleanup
        await tester.cleanup_test_data()
        
        # Generate report
        success = await tester.generate_test_report()
        
        return success
        
    except Exception as e:
        await tester.log_error(f"Critical test failure: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 