-- ============================================================================
-- Database Initialization Script for Causal Eval Bench
-- ============================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "citext";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS causal_eval;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO causal_eval, public;

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE task_type AS ENUM (
    'causal_attribution',
    'counterfactual_reasoning', 
    'causal_intervention',
    'causal_chain',
    'confounding_analysis'
);

CREATE TYPE difficulty_level AS ENUM (
    'easy',
    'medium',
    'hard',
    'expert'
);

CREATE TYPE evaluation_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled'
);

CREATE TYPE model_type AS ENUM (
    'openai',
    'anthropic',
    'huggingface',
    'cohere',
    'custom'
);

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Models table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    model_type model_type NOT NULL,
    version VARCHAR(100),
    description TEXT,
    api_endpoint VARCHAR(500),
    config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Domains table
CREATE TABLE IF NOT EXISTS domains (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_type task_type NOT NULL,
    domain_id UUID REFERENCES domains(id),
    difficulty difficulty_level NOT NULL,
    question TEXT NOT NULL,
    context TEXT,
    ground_truth JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Evaluations table
CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id),
    task_id UUID REFERENCES tasks(id),
    status evaluation_status DEFAULT 'pending',
    response TEXT,
    score DECIMAL(5,4),
    metrics JSONB DEFAULT '{}',
    execution_time_ms INTEGER,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation sessions table  
CREATE TABLE IF NOT EXISTS evaluation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),
    model_id UUID REFERENCES models(id),
    config JSONB DEFAULT '{}',
    total_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    average_score DECIMAL(5,4),
    status evaluation_status DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Link evaluations to sessions
CREATE TABLE IF NOT EXISTS session_evaluations (
    session_id UUID REFERENCES evaluation_sessions(id) ON DELETE CASCADE,
    evaluation_id UUID REFERENCES evaluations(id) ON DELETE CASCADE,
    PRIMARY KEY (session_id, evaluation_id)
);

-- ============================================================================
-- ANALYTICS TABLES
-- ============================================================================

-- Model performance analytics
CREATE TABLE IF NOT EXISTS analytics.model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id),
    task_type task_type,
    domain_id UUID REFERENCES domains(id),
    difficulty difficulty_level,
    total_evaluations INTEGER DEFAULT 0,
    average_score DECIMAL(5,4),
    median_score DECIMAL(5,4),
    score_std_dev DECIMAL(5,4),
    average_execution_time_ms INTEGER,
    success_rate DECIMAL(5,4),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Daily aggregation
CREATE TABLE IF NOT EXISTS analytics.daily_stats (
    date DATE PRIMARY KEY,
    total_evaluations INTEGER DEFAULT 0,
    total_models INTEGER DEFAULT 0,
    total_sessions INTEGER DEFAULT 0,
    average_score DECIMAL(5,4),
    total_execution_time_ms BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- AUDIT TABLES
-- ============================================================================

-- Audit log for all table changes
CREATE TABLE IF NOT EXISTS audit.activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(10) NOT NULL,
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_evaluations_model_id ON evaluations(model_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_task_id ON evaluations(task_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_score ON evaluations(score);

CREATE INDEX IF NOT EXISTS idx_tasks_task_type ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_domain_id ON tasks(domain_id);
CREATE INDEX IF NOT EXISTS idx_tasks_difficulty ON tasks(difficulty);

CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);

CREATE INDEX IF NOT EXISTS idx_evaluation_sessions_model_id ON evaluation_sessions(model_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_sessions_status ON evaluation_sessions(status);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_evaluations_model_task ON evaluations(model_id, task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_type_domain ON tasks(task_type, domain_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_model_status ON evaluations(model_id, status);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_tasks_question_fulltext ON tasks USING gin(to_tsvector('english', question));
CREATE INDEX IF NOT EXISTS idx_tasks_context_fulltext ON tasks USING gin(to_tsvector('english', context));

-- GIN indexes for JSONB fields
CREATE INDEX IF NOT EXISTS idx_tasks_metadata_gin ON tasks USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_evaluations_metrics_gin ON evaluations USING gin(metrics);
CREATE INDEX IF NOT EXISTS idx_models_config_gin ON models USING gin(config);

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to models table
CREATE TRIGGER update_models_updated_at 
    BEFORE UPDATE ON models 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit.activity_log (table_name, operation, record_id, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit.activity_log (table_name, operation, record_id, old_values, new_values)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit.activity_log (table_name, operation, record_id, old_values)
        VALUES (TG_TABLE_NAME, TG_OP, OLD.id, row_to_json(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Apply audit triggers
CREATE TRIGGER audit_models_trigger
    AFTER INSERT OR UPDATE OR DELETE ON models
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_evaluations_trigger
    AFTER INSERT OR UPDATE OR DELETE ON evaluations
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Function to calculate model performance metrics
CREATE OR REPLACE FUNCTION calculate_model_performance(
    p_model_id UUID,
    p_task_type task_type DEFAULT NULL,
    p_domain_id UUID DEFAULT NULL
)
RETURNS TABLE (
    avg_score DECIMAL(5,4),
    median_score DECIMAL(5,4),
    total_evaluations INTEGER,
    success_rate DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(e.score)::DECIMAL(5,4) as avg_score,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY e.score)::DECIMAL(5,4) as median_score,
        COUNT(*)::INTEGER as total_evaluations,
        (COUNT(*) FILTER (WHERE e.status = 'completed')::DECIMAL / COUNT(*))::DECIMAL(5,4) as success_rate
    FROM evaluations e
    JOIN tasks t ON e.task_id = t.id
    WHERE e.model_id = p_model_id
        AND (p_task_type IS NULL OR t.task_type = p_task_type)
        AND (p_domain_id IS NULL OR t.domain_id = p_domain_id)
        AND e.status = 'completed';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default domains
INSERT INTO domains (name, description) VALUES
    ('medical', 'Medical and healthcare causal reasoning'),
    ('economic', 'Economic and financial causal relationships'),
    ('social', 'Social science and human behavior causation'),
    ('scientific', 'Natural science causal phenomena'),
    ('legal', 'Legal causation and liability'),
    ('environmental', 'Environmental and ecological causation')
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Model leaderboard view
CREATE OR REPLACE VIEW model_leaderboard AS
SELECT 
    m.name as model_name,
    m.model_type,
    COUNT(e.id) as total_evaluations,
    AVG(e.score) as average_score,
    STDDEV(e.score) as score_stddev,
    COUNT(*) FILTER (WHERE e.status = 'completed') as completed_evaluations,
    COUNT(*) FILTER (WHERE e.status = 'failed') as failed_evaluations,
    AVG(e.execution_time_ms) as avg_execution_time_ms,
    MAX(e.completed_at) as last_evaluation
FROM models m
LEFT JOIN evaluations e ON m.id = e.model_id
WHERE m.is_active = true
GROUP BY m.id, m.name, m.model_type
ORDER BY average_score DESC NULLS LAST;

-- Task difficulty distribution view
CREATE OR REPLACE VIEW task_difficulty_stats AS
SELECT 
    t.task_type,
    t.difficulty,
    d.name as domain_name,
    COUNT(*) as task_count,
    AVG(e.score) as average_score,
    COUNT(e.id) as total_evaluations
FROM tasks t
LEFT JOIN domains d ON t.domain_id = d.id
LEFT JOIN evaluations e ON t.id = e.task_id AND e.status = 'completed'
GROUP BY t.task_type, t.difficulty, d.name
ORDER BY t.task_type, t.difficulty;

-- Recent evaluation activity view
CREATE OR REPLACE VIEW recent_activity AS
SELECT 
    e.id,
    m.name as model_name,
    t.task_type,
    d.name as domain_name,
    t.difficulty,
    e.score,
    e.status,
    e.execution_time_ms,
    e.created_at
FROM evaluations e
JOIN models m ON e.model_id = m.id
JOIN tasks t ON e.task_id = t.id
LEFT JOIN domains d ON t.domain_id = d.id
ORDER BY e.created_at DESC
LIMIT 100;

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

-- Create application user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'causal_eval_app') THEN
        CREATE ROLE causal_eval_app LOGIN PASSWORD 'secure_app_password';
    END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA causal_eval TO causal_eval_app;
GRANT USAGE ON SCHEMA analytics TO causal_eval_app;
GRANT SELECT ON SCHEMA audit TO causal_eval_app;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA causal_eval TO causal_eval_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA analytics TO causal_eval_app;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO causal_eval_app;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA causal_eval TO causal_eval_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO causal_eval_app;

-- Grant permissions on views
GRANT SELECT ON model_leaderboard TO causal_eval_app;
GRANT SELECT ON task_difficulty_stats TO causal_eval_app;
GRANT SELECT ON recent_activity TO causal_eval_app;

-- ============================================================================
-- COMPLETION
-- ============================================================================

-- Log initialization completion
INSERT INTO audit.activity_log (table_name, operation, new_values) 
VALUES ('system', 'INIT', '{"message": "Database initialization completed successfully"}');

COMMIT;