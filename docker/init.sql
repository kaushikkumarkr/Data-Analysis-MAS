-- Initialize pgvector extension and create tables for semantic memory

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create memories table
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- MiniLM-L6-v2 produces 384-dim embeddings
    memory_type VARCHAR(50) NOT NULL DEFAULT 'fact',
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS memories_embedding_idx 
ON memories USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for filtering by type
CREATE INDEX IF NOT EXISTS memories_type_idx ON memories(memory_type);

-- Create index for filtering by user
CREATE INDEX IF NOT EXISTS memories_user_idx ON memories(user_id);

-- Create index for filtering by session
CREATE INDEX IF NOT EXISTS memories_session_idx ON memories(session_id);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update timestamp
DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
