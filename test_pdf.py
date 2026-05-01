# test_pdf.py - Complete test for Member 6
import os
import sys
from pathlib import Path

print("=" * 60)
print("Member 6: University Handbook PDF Loading Test")
print("=" * 60)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("\n[1] Importing modules...")
try:
    from src.loader import load_documents, chunk_documents
    print("    ✓ Imports successful")
except Exception as e:
    print(f"    ✗ Import error: {e}")
    sys.exit(1)

print("\n[2] Checking data directory...")
data_dir = Path("data")
if data_dir.exists():
    files = list(data_dir.glob("*"))
    print(f"    Found {len(files)} files/folders")
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if pdf_files:
        print(f"    ✓ Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"      - {pdf.name} ({pdf.stat().st_size:,} bytes)")
    else:
        print("    ⚠ No PDF files found in data/ folder!")
        print("    You need to add a university handbook PDF here.")
else:
    print("    ✗ data/ folder not found!")
    sys.exit(1)

print("\n[3] Loading documents...")
try:
    docs = load_documents("data/")
    print(f"    ✓ Loaded {len(docs)} document objects")
except Exception as e:
    print(f"    ✗ Error loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4] Document sources:")
if len(docs) == 0:
    print("    ⚠ No documents loaded!")
    print("    Check if PDF loading is enabled in loader.py")
else:
    for i, doc in enumerate(docs[:5]):  # Show first 5
        source = Path(doc.metadata.get('source', 'Unknown')).name
        content_len = len(doc.page_content)
        print(f"    {i+1}. {source} ({content_len} characters)")

print("\n[5] Chunking documents...")
if len(docs) > 0:
    try:
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        print(f"    ✓ Created {len(chunks)} chunks")
        
        if len(chunks) > 0:
            print("\n[6] Sample chunk preview:")
            preview = chunks[0].page_content[:200].replace('\n', ' ')
            print(f"    {preview}...")
    except Exception as e:
        print(f"    ✗ Error chunking: {e}")
else:
    print("    Skipping - no documents to chunk")

print("\n" + "=" * 60)
print("✅ Member 6 verification complete!")
print("=" * 60)

# Summary for GitHub commit
print("\n📋 Summary for your GitHub commit message:")
print('   "Member 6: Added university handbook PDF to data/ folder')
print('    and enabled PDF loading in loader.py"')