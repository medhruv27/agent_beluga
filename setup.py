"""
Setup script for the GenAI Chatbot
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_docker():
    """Check if Docker is installed and running"""
    print("🐳 Checking Docker...")
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is installed")
            
            # Check if Docker daemon is running
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                return True
            else:
                print("❌ Docker daemon is not running")
                return False
        else:
            print("❌ Docker is not installed")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False


def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")


def install_dependencies():
    """Install Python dependencies"""
    # Determine the correct pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python dependencies")


def setup_environment_file():
    """Setup environment file"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        print("📝 Creating .env file from template...")
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ .env file created. Please edit it with your API keys.")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("❌ .env.example file not found")
        return False


def start_services():
    """Start required services with Docker"""
    print("🚀 Starting required services...")
    
    # Start Qdrant
    qdrant_cmd = "docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest"
    if not run_command(qdrant_cmd, "Starting Qdrant"):
        return False
    
    # Start Redis
    redis_cmd = "docker run -d --name redis -p 6379:6379 redis:7-alpine"
    if not run_command(redis_cmd, "Starting Redis"):
        return False
    
    print("✅ All services started successfully")
    return True


def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    
    # Determine the correct python command based on OS
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    # Test imports
    test_script = """
import sys
try:
    import fastapi
    import streamlit
    import qdrant_client
    import redis
    import langchain
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    return run_command(f'{python_cmd} -c "{test_script}"', "Testing package imports")


def main():
    """Main setup function"""
    print("🤖 GenAI Chatbot Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_docker():
        print("⚠️  Docker is required for running Qdrant and Redis")
        print("   Please install Docker and try again")
        sys.exit(1)
    
    # Setup Python environment
    if not create_virtual_environment():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    # Setup configuration
    if not setup_environment_file():
        sys.exit(1)
    
    # Start services
    if not start_services():
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    print("3. Start the application:")
    print("   python main.py")
    print("4. Open Streamlit frontend:")
    print("   streamlit run streamlit_app.py")
    print("\n🌐 Access the application at:")
    print("   - Backend API: http://localhost:8000")
    print("   - Frontend UI: http://localhost:8501")
    print("   - Qdrant: http://localhost:6333")
    print("   - Redis: localhost:6379")


if __name__ == "__main__":
    main()
