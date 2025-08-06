# Contextual Retrieval Implementation

This directory contains the implementation of contextual retrieval functionality that enhances document chunks with contextual information using the specified prompt.

## Overview

The contextual retrieval feature enhances document chunks by adding contextual information at the top of each document's `page_content`. This is done using the specified prompt with the exaone3.5 model to generate contextual information that helps situate each chunk within the overall document.

## Files

- `contextual_retrieval_test.ipynb`: Jupyter notebook for testing and experimenting with the contextual retrieval functionality
- `contextual_retrieval.py`: Python script version for easier execution and testing
- `README.md`: This documentation file

## Implementation Details

### Contextual Prompt

The implementation uses the following prompt to generate contextual information:

```
<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

### Key Components

1. **ContextualRetrieval Class**: Main class that handles the contextual enhancement process
2. **Document Enhancement**: Each document is enhanced with contextual information at the top
3. **Vector Database Creation**: Enhanced documents are used to create a new FAISS vector database
4. **Retrieval Testing**: Comparison between original and enhanced retrieval results

### Usage

#### Running the Python Script

```bash
cd simul_10
python contextual_retrieval.py
```

#### Using the Jupyter Notebook

1. Open `contextual_retrieval_test.ipynb` in Jupyter
2. Run the cells sequentially to:
   - Load and split documents
   - Initialize contextual retrieval
   - Test with sample documents
   - Create enhanced vector database
   - Compare retrieval results

### Features

1. **Document Loading**: Loads markdown documents and splits them hierarchically
2. **Context Generation**: Uses exaone3.5 model to generate contextual information for each chunk
3. **Document Enhancement**: Adds contextual information to the top of each document
4. **Vector Database**: Creates FAISS vector database with enhanced documents
5. **Retrieval Testing**: Tests retrieval with enhanced documents and compares with original

### Output

The enhanced documents have the following structure:

```
[Context]: {generated_contextual_information}

[section_path]: {original_section_path}

{original_document_content}
```

### Model Configuration

- **LLM Model**: exaone3.5:latest (for context generation)
- **Embedding Model**: bge-m3:latest (for vector database)
- **Temperature**: 0.1 (for consistent context generation)

### Testing Queries

The implementation includes test queries such as:
- "원스토어 인앱결제의 PNS의 개념을 설명해주세요"
- "PNS 메시지 규격의 purchaseState는 어떤 값으로 구성되나요?"
- "원스토어 인앱결제 SDK 사용법"
- "결제 테스트 및 보안 관련 정보"

### Expected Benefits

1. **Improved Retrieval**: Enhanced documents should provide better search results
2. **Context Awareness**: Each chunk is better situated within the overall document
3. **Better Relevance**: Contextual information helps improve search relevance
4. **Enhanced Understanding**: Users get better context for retrieved documents

## Requirements

- Python 3.8+
- LangChain
- Ollama (with exaone3.5 and bge-m3 models)
- FAISS
- Jupyter (for notebook usage)

## Notes

- The implementation processes documents sequentially to avoid overwhelming the LLM
- Context generation may take time for large document sets
- Original document content is preserved in metadata for comparison
- The enhanced vector database is saved separately from the original 