#!/usr/bin/env python3
"""
ARCHIVE DOCUMENT UPLOADER
Uploads all documents from the archive directory to the knowledge base service
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentUploader:
    def __init__(self):
        self.knowledge_base_url = "http://localhost:8002"
        self.archive_path = Path("archive/sample_data")
        self.uploaded_docs = []
        self.failed_uploads = []
        
    def load_json_documents(self, json_file_path: str) -> List[Dict]:
        """Load documents from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} documents from {json_file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load {json_file_path}: {e}")
            return []
    
    def scan_directory_files(self, directory_path: Path) -> List[Dict]:
        """Scan directory for individual text/markdown files"""
        documents = []
        
        # Map directory names to categories
        category_mapping = {
            'faqs': ('Support', 'FAQ'),
            'policies': ('Support', 'Policies'),
            'product_info': ('Product', 'Information'),
            'troubleshooting': ('Support', 'Troubleshooting'),
            'procedures': ('Support', 'Procedures'),
            'shipping': ('Support', 'Shipping'),
            'billing': ('Support', 'Billing'),
            'returns': ('Support', 'Returns')
        }
        
        for category_dir in directory_path.iterdir():
            if category_dir.is_dir() and category_dir.name in category_mapping:
                category, subcategory = category_mapping[category_dir.name]
                
                for file_path in category_dir.glob('*'):
                    if file_path.suffix in ['.txt', '.md'] and file_path.is_file():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Create document structure
                            doc = {
                                "title": file_path.stem.replace('_', ' ').title(),
                                "content": content,
                                "category": category,
                                "subcategory": subcategory,
                                "tags": [category_dir.name, file_path.stem.replace('_', '-')],
                                "metadata": {
                                    "source_file": str(file_path),
                                    "file_type": file_path.suffix[1:],
                                    "category_dir": category_dir.name
                                }
                            }
                            documents.append(doc)
                            
                        except Exception as e:
                            logger.error(f"Failed to read {file_path}: {e}")
        
        logger.info(f"Scanned {len(documents)} files from directory structure")
        return documents
    
    def upload_document(self, document: Dict) -> bool:
        """Upload a single document to the knowledge base service"""
        try:
            # Ensure required fields are present
            required_fields = ['title', 'content', 'category']
            for field in required_fields:
                if field not in document:
                    logger.error(f"Document missing required field: {field}")
                    return False
            
            # Clean up the document structure for API
            upload_doc = {
                "title": document['title'],
                "content": document['content'],
                "category": document['category'],
                "subcategory": document.get('subcategory', ''),
                "tags": document.get('tags', []),
                "metadata": document.get('metadata', {})
            }
            
            # Add upload timestamp
            upload_doc['metadata']['upload_timestamp'] = time.time()
            upload_doc['metadata']['upload_source'] = 'archive_batch_upload'
            
            response = requests.post(
                f"{self.knowledge_base_url}/api/v1/documents",
                json=upload_doc,
                timeout=30
            )
            
            if response.status_code == 200:
                doc_response = response.json()
                doc_id = doc_response.get('id', 'unknown')
                logger.info(f"✅ Uploaded: {document['title']} (ID: {doc_id})")
                self.uploaded_docs.append({
                    'title': document['title'],
                    'id': doc_id,
                    'category': document['category']
                })
                return True
            else:
                logger.error(f"❌ Upload failed for {document['title']}: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.failed_uploads.append(document['title'])
                return False
                
        except Exception as e:
            logger.error(f"❌ Upload error for {document['title']}: {e}")
            self.failed_uploads.append(document['title'])
            return False
    
    def upload_all_documents(self):
        """Upload all documents from archive directory"""
        logger.info("🚀 Starting Archive Document Upload Process")
        logger.info("="*60)
        
        all_documents = []
        
        # Load documents from JSON files
        json_files = [
            self.archive_path / "sample_support_documents.json",
            self.archive_path / "sample_documents.json"
        ]
        
        for json_file in json_files:
            if json_file.exists():
                json_docs = self.load_json_documents(json_file)
                all_documents.extend(json_docs)
        
        # Load documents from directory structure
        support_docs_dir = self.archive_path / "sample_support_documents"
        if support_docs_dir.exists():
            dir_docs = self.scan_directory_files(support_docs_dir)
            all_documents.extend(dir_docs)
        
        # Remove duplicates based on title
        unique_docs = {}
        for doc in all_documents:
            title = doc['title']
            if title not in unique_docs:
                unique_docs[title] = doc
            else:
                # Keep the one with more content
                if len(doc.get('content', '')) > len(unique_docs[title].get('content', '')):
                    unique_docs[title] = doc
        
        final_documents = list(unique_docs.values())
        
        logger.info(f"📚 Total unique documents to upload: {len(final_documents)}")
        logger.info("="*60)
        
        # Upload documents with progress
        success_count = 0
        for i, document in enumerate(final_documents, 1):
            logger.info(f"📄 ({i}/{len(final_documents)}) Uploading: {document['title']}")
            
            if self.upload_document(document):
                success_count += 1
            
            # Small delay to avoid overwhelming the service
            time.sleep(0.5)
        
        # Generate summary report
        self.generate_upload_report(len(final_documents), success_count)
    
    def generate_upload_report(self, total_docs: int, success_count: int):
        """Generate upload summary report"""
        logger.info("="*60)
        logger.info("📊 DOCUMENT UPLOAD SUMMARY REPORT")
        logger.info("="*60)
        
        success_rate = (success_count / total_docs) * 100 if total_docs > 0 else 0
        
        logger.info(f"Total Documents Processed: {total_docs}")
        logger.info(f"Successfully Uploaded: {success_count}")
        logger.info(f"Failed Uploads: {len(self.failed_uploads)}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.uploaded_docs:
            logger.info(f"\n📚 Categories Uploaded:")
            categories = {}
            for doc in self.uploaded_docs:
                category = doc['category']
                categories[category] = categories.get(category, 0) + 1
            
            for category, count in categories.items():
                logger.info(f"   {category}: {count} documents")
        
        if self.failed_uploads:
            logger.info(f"\n❌ Failed Uploads:")
            for title in self.failed_uploads[:5]:  # Show first 5
                logger.info(f"   - {title}")
            if len(self.failed_uploads) > 5:
                logger.info(f"   ... and {len(self.failed_uploads) - 5} more")
        
        logger.info("="*60)
        
        if success_rate >= 80:
            logger.info("🎉 DOCUMENT UPLOAD COMPLETED SUCCESSFULLY!")
            logger.info("Your knowledge base is now populated with comprehensive support documents.")
        elif success_rate >= 60:
            logger.info("⚠️  DOCUMENT UPLOAD MOSTLY SUCCESSFUL - Some issues encountered")
        else:
            logger.info("❌ DOCUMENT UPLOAD HAD SIGNIFICANT ISSUES - Check service status")
        
        logger.info("\n🔍 You can now test the system with queries like:")
        logger.info("   - 'What is your warranty policy?'")
        logger.info("   - 'How do I process a return?'")
        logger.info("   - 'What payment methods do you accept?'")
        logger.info("   - 'How do I troubleshoot laptop issues?'")
        
        return success_rate >= 80

def main():
    """Main function to run the document upload process"""
    uploader = DocumentUploader()
    
    # Check if knowledge base service is available
    try:
        response = requests.get(f"{uploader.knowledge_base_url}/health", timeout=10)
        if response.status_code != 200:
            logger.error("❌ Knowledge Base Service is not available. Please start the services first.")
            logger.error("Run: cd setup && docker-compose up -d")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to Knowledge Base Service: {e}")
        logger.error("Run: cd setup && docker-compose up -d")
        return False
    
    # Start upload process
    success = uploader.upload_all_documents()
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 