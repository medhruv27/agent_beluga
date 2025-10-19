#!/usr/bin/env python3
"""
Startup script for the GenAI Chatbot
"""
import subprocess
import sys
import os
import time
import requests
from pathlib import Path


def check_services():
    """Check if required services are running"""
    services = {
        "Qdrant": "http://localhost:6333",
        "Redis": "redis://localhost:6379"
    }
    
    print("🔍 Checking required services...")
    
    # Check Qdrant
    try:
        response = requests.get("http://localhost:6333/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running")
        else:
            print("❌ Qdrant is not responding properly")
            return False
    except requests.exceptions.RequestException:
        print("❌ Qdrant is not running")
        return False
    
    # Check Redis (simplified check)
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis is running")
    except Exception:
        print("❌ Redis is not running")
        return False
    
    return True


def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    
    try:
        # Start backend in background
        backend_process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if backend is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend is running on http://localhost:8000")
                return backend_process
            else:
                print("❌ Backend failed to start properly")
                return None
        except requests.exceptions.RequestException:
            print("❌ Backend is not responding")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None


def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    
    try:
        # Start frontend in background
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(5)
        
        # Check if frontend is running
        try:
            response = requests.get("http://localhost:8501", timeout=10)
            if response.status_code == 200:
                print("✅ Frontend is running on http://localhost:8501")
                return frontend_process
            else:
                print("❌ Frontend failed to start properly")
                return None
        except requests.exceptions.RequestException:
            print("❌ Frontend is not responding")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None


def load_sample_data():
    """Load sample data into the system"""
    print("📚 Loading sample data...")
    
    try:
        from sample_data import SAMPLE_DOCUMENTS
        from agents import RetrievalAgent
        
        # Initialize retrieval agent
        retrieval_agent = RetrievalAgent()
        
        # Add sample documents
        success = asyncio.run(retrieval_agent.add_documents(SAMPLE_DOCUMENTS))
        
        if success:
            print("✅ Sample data loaded successfully")
        else:
            print("⚠️  Sample data loading failed, but system will still work")
            
    except Exception as e:
        print(f"⚠️  Could not load sample data: {e}")
        print("   System will work with empty knowledge base")


def main():
    """Main startup function"""
    print("🤖 GenAI Chatbot Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Please run this script from the chatbot directory")
        sys.exit(1)
    
    # Check required services
    if not check_services():
        print("\n❌ Required services are not running.")
        print("Please start Qdrant and Redis first:")
        print("  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest")
        print("  docker run -d --name redis -p 6379:6379 redis:7-alpine")
        sys.exit(1)
    
    # Load sample data
    load_sample_data()
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n🎉 GenAI Chatbot is now running!")
    print("\n📋 Access Points:")
    print("  • Frontend UI: http://localhost:8501")
    print("  • Backend API: http://localhost:8000")
    print("  • API Docs: http://localhost:8000/docs")
    print("  • Qdrant: http://localhost:6333")
    print("\n💡 Tips:")
    print("  • Ask Sophia about Generative AI topics")
    print("  • She will ask clarifying questions to help you better")
    print("  • Check the sidebar for evaluation metrics")
    print("  • Press Ctrl+C to stop the application")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Terminate processes
        if frontend_process:
            frontend_process.terminate()
        if backend_process:
            backend_process.terminate()
        
        print("✅ Application stopped")


if __name__ == "__main__":
    import asyncio
    main()
