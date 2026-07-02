> **Archived design:** these diagrams describe the removed FAISS implementation. See
> [README.md](README.md) for the current Qdrant architecture.

# Document Flow - Legacy FAISS Visual Guide

## 🔄 How Documents Flow Through the System

### **Scenario 1: System Startup**

```
┌─────────────────────────────────────────────────┐
│         SYSTEM STARTUP                          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  get_retriever() called │
        │  (Initialization)       │
        └────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │  Create DUMMY Document             │
    │  - Content: "Placeholder..."       │
    │  - Purpose: Init FAISS vectorstore │
    └────────────┬─────────��───────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │  Initialize FAISS Vectorstore      │
    │  Documents in store: 1 (dummy)     │
    │  Status: ✅ Ready                  │
    └────────────────────────────────────┘
```

---

### **Scenario 2: User Uploads Document**

```
┌────────────────────────────────────────────┐
│  USER UPLOADS DOCUMENT                     │
│  POST /rag/documents/upload                │
│  File: "Python_Guide.pdf"                  │
│  Description: "Python tutorial"            │
└──────────────┬─────────────────────────────┘
               │
               ▼
        ┌─────────────────────┐
        │  Load PDF File      │
        │  Extract content    │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Split into Chunks  │
        │  Chunk Size: 1000   │
        │  Output: 20 chunks  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  retriever_chain(chunks)     │
        │  Store in FAISS vectorstore  │
        │  (ACTUAL DOCUMENTS STORED)   │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  FAISS Update Complete       │
        │  Documents in store: 21      │
        │  - 1 dummy (init)            │
        │  - 20 real document chunks   │
        │  Status: ✅ Ready            │
        └─────────────────────────���────┘
```

---

### **Scenario 3: User Queries Document**

```
┌────────────────────────────────────────────┐
│  USER ASKS QUESTION                        │
│  POST /rag/query                           │
│  Query: "What is Python?"                  │
└──────────────┬──���──────────────────────────┘
               │
               ▼
        ┌─────────────────────────────┐
        │  Query Router               │
        │  - Classify query type      │
        │  - Route to appropriate     │
        │    pipeline                 │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │  get_retriever() called     │
        │  - Get FAISS retriever      │
        │  - Returns tool             │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  FAISS Search                │
        │  Search query: "What is...?" │
        │                              │
        │  Search FAISS vectorstore:   │
        │  ✅ Find matching chunks     │
        │     from Python_Guide.pdf   │
        │ ❌ Ignore dummy placeholder  │
        │                              │
        │  Results: 3-5 relevant       │
        │  document chunks             │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌─────────────────���────────────┐
        │  Pass to LLM                 │
        │  - Context: Found chunks     │
        │  - Question: User query      │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌─────────────────────���────────┐
        │  Generate Answer             │
        │  Answer based on YOUR docs   │
        │  "Python is a programming..." │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  Return to User              │
        │  Answer: "Python is..."      │
        │  Status: ✅ Complete         │
        └──────────────────────────────┘
```

---

### **Scenario 4: Upload Second Document**

```
┌────────────────────────────────��───────┐
│  USER UPLOADS ANOTHER DOCUMENT          │
│  File: "Web_Dev_Guide.pdf"              │
└────────────┬───────────────────────────┘
             │
             ▼
    ┌──────────────────────┐
    │  Load & Process      │
    │  Split into chunks   │
    │  Output: 15 chunks   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  retriever_chain(chunks)     │
    │  Add to existing FAISS       │
    │  (Now has both docs!)        │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  FAISS Update Complete       │
    │  Documents in store: 36      │
    │  - 1 dummy (init)            │
    │  - 20 chunks (Python guide)  │
    │  - 15 chunks (Web dev guide) │
    │  Status: ✅ Ready            │
    │                              │
    │  Can search BOTH documents   │
    │  with a single query!        │
    └──────────────────────────────┘
```

---

## 📊 Document Storage State

### **Timeline**

```
TIME 1: System Startup
┌─────────────────────────┐
│ FAISS Vectorstore       │
├─────────────────────────┤
│ ✓ Dummy (1)             │
│                         │
│ Total: 1 document       │
└─────────────────────────┘

TIME 2: Upload Python Guide
┌──────────────────────��──┐
│ FAISS Vectorstore       │
├─────────────────────────┤
│ ✓ Dummy (1)             │
│ ✓ Python Guide (20)     │
│                         │
│ Total: 21 documents     │
└─────────────────────────┘

TIME 3: Query
┌─────────────────────────┐
│ Search Results:         │
├─────────────────────────┤
│ Found in:               │
│ ✓ Python Guide chunks   │
│ ✗ Dummy (not relevant)  │
│                         │
│ Return: Relevant chunks │
└─────────────────────────┘

TIME 4: Upload Web Dev Guide
┌─────────────────────────┐
│ FAISS Vectorstore       │
├─────────────────────────┤
│ ✓ Dummy (1)             │
│ ✓ Python Guide (20)     │
│ ✓ Web Dev Guide (15)    │
│                         │
│ Total: 36 documents     │
└─────────────────────────┘

TIME 5: Query Again
┌─────────────���────────────┐
│ Can search BOTH docs:    │
├──────────────────────────┤
│ Search: "web development"│
│ Found in:                │
│ ✓ Web Dev Guide chunks   │
│ ✓ Python Guide (if match)│
│ ✗ Dummy                  │
│                          │
│ Return: All matches      │
└──────────────────────────┘
```

---

## 🔑 Key Takeaways

```
┌─────────────────────────────────────────────────┐
│ IMPORTANT FLOW                                  │
├─────────────────────────────────────────────────┤
│                                                 │
│ 1. Startup:                                     │
│    Create dummy → Vectorstore ready             │
│                                                 │
│ 2. Upload Document:                             │
│    Load → Split → Store in FAISS → Ready       │
│                                                 │
│ 3. Query:                                       │
│    Search FAISS → Find YOUR docs → LLM answer  │
│                                                 │
│ 4. Upload Another:                              │
│    Add more → FAISS has both → Search both      │
│                                                 │
│ RESULT: ✅ YOUR DOCUMENTS ARE STORED & USED    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Ready to Test!

You can now:

1. **Start the system**
   ```bash
   python -m uvicorn src.main:app --reload
   ```

2. **Upload your document**
   ```bash
   curl -X POST http://localhost:8000/rag/documents/upload \
     -H "X-Description: My Document" \
     -F "file=@my_file.pdf"
   ```

3. **Query it**
   ```bash
   curl -X POST http://localhost:8000/rag/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Your question", "session_id": "user_1"}'
   ```

4. **Get answers from YOUR documents!** ✅

---

**Status**: ✅ System works correctly  
**Your documents**: Will be found and used  
**Ready to use**: YES!

