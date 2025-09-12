# Main Pipeline Documentation

This document explains how to use the `mainPipeline.py` functionality for processing social media posts through the AI text analysis system.

## Overview

The pipeline connects to a dump database (twitter_scrapper) and processes posts from the `post_bank` table through the `/api/process-text` endpoint, saving the analysis results to an `analyzed_Data` table.

## Database Structure

### Source Table: `post_bank`
- Contains social media posts with fields like `post_snippet`, `post_title`, `source`, etc.
- New column `analysisStatus` (default: `NOT_ANALYZED`)
- Gets updated to `ANALYZED` after processing

### Target Table: `analyzed_Data`
- Stores the 28-field response from `/api/process-text`
- Contains `dump_table_id` to reference the original `post_bank` row
- Includes all analysis results: entities, sentiment, topics, etc.

## Configuration

The pipeline uses environment variables from `.env`:

```env
# DUMP DATABASE Configuration
DUMP_DB_HOST=94.136.189.147
DUMP_DB_NAME=twitter_scrapper
DUMP_DB_USER=gccloud
DUMP_DB_PASSWORD=Gccloud@1489$
DUMP_DB_PORT=3306

# API Configuration
API_BASE_URL=http://localhost:8000
```

## Usage

### 1. Basic Pipeline Execution

```python
from mainPipeline import PipelineProcessor

# Run pipeline with default settings
with PipelineProcessor() as processor:
    results = processor.run_pipeline()
    print(f"Processed: {results}")
```

### 2. Batch Processing

```python
# Process in smaller batches
with PipelineProcessor() as processor:
    results = processor.run_pipeline(batch_size=10)
    print(f"Batch results: {results}")
```

### 3. Get Statistics

```python
# Check pipeline status
with PipelineProcessor() as processor:
    stats = processor.get_pipeline_stats()
    print(f"Unanalyzed posts: {stats['unanalyzed_posts']}")
    print(f"Total analyzed: {stats['analyzed_posts']}")
```

### 4. Manual Table Creation

```python
# Create/verify tables
with PipelineProcessor() as processor:
    success = processor.create_tables()
    if success:
        print("Tables ready!")
```

## Command Line Usage

### Run Tests
```bash
python test_pipeline.py
```

### Run Pipeline Directly
```bash
python mainPipeline.py
```

## Pipeline Flow

1. **Connect** to dump database (twitter_scrapper)
2. **Create/Verify** tables (`analyzed_Data`, update `post_bank`)
3. **Query** unanalyzed posts: `SELECT * FROM post_bank WHERE analysisStatus = 'NOT_ANALYZED'`
4. **Process** each post:
   - Extract `post_snippet` text
   - Call `/api/process-text` endpoint
   - Parse 28-field response
   - Save to `analyzed_Data` table
   - Update `analysisStatus` to `ANALYZED`
5. **Repeat** until no unanalyzed posts remain

## Response Fields (28 total)

The API returns comprehensive analysis including:

- **Text Processing**: `input_text`, `processed_text`, `enhanced_text`
- **Language**: `detected_language`, `language_confidence`
- **Topics**: `topic_id`, `topic_title`, `topic_description`, `confidence`
- **Entities**: `person_names`, `organisation_names`, `location_names`, etc.
- **Sentiment**: `sentiment_label`, `sentiment_score`, `emotion_primary`
- **Metadata**: `processing_time_ms`, `model_version`, `boost_reasons`

## Error Handling

- **Connection Errors**: Automatic retry with exponential backoff
- **API Failures**: Skip problematic posts, continue processing
- **Database Errors**: Transaction rollback, detailed logging
- **Empty Results**: Graceful handling with status messages

## Performance Features

- **Batch Processing**: Configurable batch sizes
- **Connection Pooling**: Efficient database connections
- **Progress Tracking**: Real-time statistics and logging
- **Memory Management**: Processes data in chunks
- **Recursive Processing**: Continues until all posts analyzed

## Monitoring

```python
# Check progress
with PipelineProcessor() as processor:
    stats = processor.get_pipeline_stats()
    
    if stats['unanalyzed_posts'] == 0:
        print("✅ All posts analyzed!")
    else:
        print(f"📊 {stats['unanalyzed_posts']} posts remaining")
```

## Troubleshooting

### Common Issues

1. **Connection Failed**: Check database credentials in `.env`
2. **API Timeout**: Ensure the main application is running on port 8000
3. **Table Not Found**: Run `processor.create_tables()` first
4. **Memory Issues**: Reduce `batch_size` parameter

### Logs

The pipeline provides detailed logging:
- Database operations
- API call results
- Processing statistics
- Error details

## Example Output

```
🚀 Starting Pipeline Processing
📊 Found 150 unanalyzed posts
🔄 Processing batch 1/15 (10 posts)
✅ Batch 1 completed: 10 processed, 0 failed
🔄 Processing batch 2/15 (10 posts)
✅ Batch 2 completed: 10 processed, 0 failed
...
✅ Pipeline completed successfully!
📈 Final stats: 150 analyzed, 0 failed, 0 remaining
```

## Integration

The pipeline is designed to work alongside the existing AI text analyzer without disrupting current functionality. It operates on the separate dump database and can be run independently or scheduled as needed.