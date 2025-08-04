# data/embed_data.py
"""
Enhanced script to embed self_data.json with tone metadata and upload to Pinecone
Now includes tone information in metadata for better response customization
"""

import json
import os
import sys
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import validate_env_vars

def main():
    """Main function to embed and upload data to Pinecone with tone metadata"""
    
    print("🚀 Starting AStarBot data embedding process with tone support...")
    
    # Load environment variables from parent directory
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    
    # Validate required environment variables
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    env_status = validate_env_vars(required_vars)
    
    missing_vars = [var for var, exists in env_status.items() if not exists]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        print("\n💡 Please check your .env file in the parent directory")
        return
    
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "starbot")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
    
    # Initialize embedding model
    print("📥 Loading sentence transformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model loaded")
    
    # Initialize Pinecone
    print("🔌 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Delete existing index if it exists (fresh start)
    if INDEX_NAME in pc.list_indexes().names():
        print(f"🗑️ Deleting existing index: {INDEX_NAME}")
        pc.delete_index(INDEX_NAME)
    
    # Create new index
    print(f"🆕 Creating new index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENV)
    )
    
    # Wait for index to be ready
    index = pc.Index(INDEX_NAME)
    print("✅ Index created and ready")
    
    # Load enhanced data with tone information
    data_path = "self_data.json"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("📝 Please make sure self_data.json exists in the data/ directory")
        return
    
    print(f"📖 Loading Avrodeep's enhanced data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    print(f"📊 Found {len(entries)} entries about Avrodeep")
    
    # Analyze tone distribution
    tone_counts = {"basic": 0, "friendly": 0, "professional": 0}
    for entry in entries:
        tone = entry.get("tone", "professional")
        if tone in tone_counts:
            tone_counts[tone] += 1
    
    print("🎭 Tone distribution:")
    for tone, count in tone_counts.items():
        print(f"  • {tone}: {count} entries")
    
    # Process and upload each entry with enhanced metadata
    successful_uploads = 0
    failed_uploads = 0
    
    for entry in entries:
        try:
            # Generate embedding
            vector = model.encode(entry["text"]).tolist()
            
            # Enhanced metadata with tone information
            metadata = {
                "text": entry["text"],
                "tags": entry["tags"],
                "tone": entry.get("tone", "professional"),  # New tone field
                "entry_id": entry["id"]
            }
            
            # Upload to Pinecone with enhanced metadata
            index.upsert(vectors=[{
                "id": entry["id"],
                "values": vector,
                "metadata": metadata
            }])
            
            tone = entry.get("tone", "professional")
            print(f"✅ Uploaded [{tone}]: {entry['id']}")
            successful_uploads += 1
            
        except Exception as e:
            print(f"❌ Error uploading {entry['id']}: {e}")
            failed_uploads += 1
    
    # Final summary with tone analysis
    print("\n" + "="*50)
    print("📈 UPLOAD SUMMARY")
    print("="*50)
    print(f"✅ Successful uploads: {successful_uploads}")
    print(f"❌ Failed uploads: {failed_uploads}")
    print(f"📊 Total entries: {len(entries)}")
    print(f"🎯 Success rate: {(successful_uploads/len(entries)*100):.1f}%")
    
    print("\n🎭 TONE BREAKDOWN:")
    for tone, count in tone_counts.items():
        percentage = (count / len(entries) * 100)
        print(f"  • {tone.title()}: {count} entries ({percentage:.1f}%)")
    
    if successful_uploads > 0:
        print(f"\n🎉 Avrodeep's enhanced data successfully embedded in Pinecone!")
        print("🧠 Memory support: Redis-backed with tone-aware responses")
        print("💡 You can now start the FastAPI server with: python main.py")
        print("🔥 Or test in CLI with: python app.py")
        print("\n🔧 Next steps:")
        print("  1. Make sure Redis is running: redis-server")
        print("  2. Update your .env with REDIS_URL=redis://localhost:6379")
        print("  3. Test different tones by asking various questions!")
    else:
        print("\n😞 No data was uploaded. Please check your data file and try again.")

if __name__ == "__main__":
    main()
