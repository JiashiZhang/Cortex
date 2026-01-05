import sqlite3
import uuid
import datetime
import os
import json
from typing import List, Dict, Any, Optional

import chromadb
# Switched to local HuggingFace embeddings to avoid API quota limits
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

# Configuration
DATA_DIR = "./data"
SQLITE_DB_PATH = os.path.join(DATA_DIR, "my_notes.db")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chromadb")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class MemoryStore:
    def __init__(self):
        self._init_sqlite()
        self._init_chroma()

    def _init_sqlite(self):
        """Initialize SQLite database for raw text storage and graph relations."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Basic notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tags TEXT
            )
        ''')
        
        # Knowledge Graph: Entities
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                type TEXT
            )
        ''')
        
        # Knowledge Graph: Relations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                note_id TEXT,
                FOREIGN KEY(source_id) REFERENCES entities(id),
                FOREIGN KEY(target_id) REFERENCES entities(id),
                FOREIGN KEY(note_id) REFERENCES notes(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _init_chroma(self):
        """Initialize ChromaDB for vector storage."""
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Use local HuggingFace model
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="personal_notes"
        )

    def add_note(self, content: str, tags: List[str] = None) -> str:
        """Save a note to both SQLite and ChromaDB."""
        note_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        tags_str = json.dumps(tags) if tags else "[]"

        # 1. Save to SQLite
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO notes (id, content, timestamp, tags) VALUES (?, ?, ?, ?)",
            (note_id, content, timestamp, tags_str)
        )
        conn.commit()
        conn.close()

        # 2. Save to ChromaDB
        embedding = self.embedding_function.embed_query(content)
        
        metadata = {"timestamp": timestamp}
        if tags:
            metadata["tags"] = ",".join(tags)

        self.collection.add(
            ids=[note_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        
        return note_id

    def add_graph_relation(self, source: str, relation: str, target: str, note_id: str = None):
        """Add a relation to the knowledge graph (Simple Triple)."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        def get_or_create_entity(name):
            cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                new_id = str(uuid.uuid4())
                cursor.execute("INSERT INTO entities (id, name, type) VALUES (?, ?, ?)", (new_id, name, "concept"))
                return new_id

        source_id = get_or_create_entity(source)
        target_id = get_or_create_entity(target)
        relation_id = str(uuid.uuid4())
        
        cursor.execute(
            "INSERT INTO relations (id, source_id, target_id, relation_type, note_id) VALUES (?, ?, ?, ?, ?)",
            (relation_id, source_id, target_id, relation, note_id)
        )
        conn.commit()
        conn.close()

    def search_notes(self, query: str, n_results: int = 5) -> str:
        """Search notes using semantic search."""
        query_embedding = self.embedding_function.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        formatted_results = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                formatted_results.append(f"[Date: {metadata.get('timestamp', 'N/A')}] {doc}")
        
        if not formatted_results:
            return "No relevant notes found."
            
        return "\n\n".join(formatted_results)

    def get_graph_context(self, entity_name: str) -> str:
        """Retrieve related entities from the knowledge graph."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Find entity ID
        cursor.execute("SELECT id FROM entities WHERE name LIKE ?", (f"%{entity_name}%",))
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return ""
        
        entity_ids = [row[0] for row in rows]
        placeholders = ','.join('?' for _ in entity_ids)
        
        # Find relations where this entity is source or target
        query = f'''
            SELECT e1.name, r.relation_type, e2.name 
            FROM relations r
            JOIN entities e1 ON r.source_id = e1.id
            JOIN entities e2 ON r.target_id = e2.id
            WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
        '''
        
        # We need to pass the list twice because of the OR condition
        cursor.execute(query, entity_ids + entity_ids)
        relations = cursor.fetchall()
        conn.close()
        
        if not relations:
            return ""
            
        context_lines = ["Knowledge Graph Connections:"]
        for src, rel, tgt in relations:
            context_lines.append(f"- {src} -> [{rel}] -> {tgt}")
            
        return "\n".join(context_lines)

# Initialize the store instance
memory = MemoryStore()

# Define Tools for LangChain Agent

@tool
def save_memory(content: str, tags: List[str] = None) -> str:
    """
    Saves a user's thought, note, diary entry, or memo to the long-term memory.
    Use this tool when the user wants to remember something or store information.
    
    Args:
        content: The text content to save.
        tags: Optional list of keywords or tags to categorize the memory (e.g., ["work", "idea"]).
    """
    return memory.add_note(content, tags)

@tool
def extract_and_save_knowledge(content: str, source_entity: str, relation: str, target_entity: str) -> str:
    """
    Extracts and saves a structured relationship (knowledge graph triple) from the content.
    Use this tool to explicitly link concepts together.
    
    Args:
        content: The original context or sentence.
        source_entity: The subject entity (e.g., "Python").
        relation: The relationship (e.g., "is a").
        target_entity: The object entity (e.g., "programming language").
    """
    # First save the raw note
    note_id = memory.add_note(content)
    # Then save the relation linked to that note
    memory.add_graph_relation(source_entity, relation, target_entity, note_id)
    return f"Saved knowledge: {source_entity} {relation} {target_entity}"

@tool
def search_memory(query: str) -> str:
    """
    Searches the long-term memory for relevant notes or information based on a query.
    It performs both semantic search and knowledge graph lookup.
    Use this tool when the user asks about past thoughts, notes, or stored information.
    """
    # 1. Semantic Search
    semantic_results = memory.search_notes(query)
    
    # 2. Knowledge Graph Context (Simple keyword extraction for demo)
    # In a real app, we'd use an LLM to extract entities from the query first
    graph_context = ""
    words = query.split()
    for word in words:
        if len(word) > 3: # Ignore short words
            context = memory.get_graph_context(word)
            if context:
                graph_context += "\n" + context
    
    final_result = semantic_results
    if graph_context:
        final_result += "\n\n" + graph_context
        
    return final_result
