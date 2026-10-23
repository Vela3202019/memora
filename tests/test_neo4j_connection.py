"""
Test script to verify Neo4j connection setup.
Run this to ensure Neo4j is properly configured before proceeding.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import logging
import pytest
from dotenv import load_dotenv
from memora.memory.client import get_neo4j_client, Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Reset singleton to ensure fresh credentials are loaded
import memora.memory.client as client_module
client_module._neo4j_client = None


async def check_async_connection():
    """Test asynchronous Neo4j connection."""
    logger.info("Testing async Neo4j connection...")
    client = get_neo4j_client()
    
    try:
        session = await client.get_async_session()
        async with session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            if record and record["test"] == 1:
                logger.info("✅ Async connection test successful!")
                return True
            else:
                logger.error("❌ Async connection test failed: Unexpected result")
                return False
    except Exception as e:
        logger.error(f"❌ Async connection test failed: {e}")
        return False
    finally:
        await client.close_async()


def check_sync_connection():
    """Test synchronous Neo4j connection."""
    logger.info("Testing sync Neo4j connection...")
    client = get_neo4j_client()
    
    try:
        with client.get_session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                logger.info("✅ Sync connection test successful!")
                return True
            else:
                logger.error("❌ Sync connection test failed: Unexpected result")
                return False
    except Exception as e:
        logger.error(f"❌ Sync connection test failed: {e}")
        return False
    finally:
        client.close()


def test_sync_connection():
    assert check_sync_connection()


@pytest.mark.asyncio
async def test_async_connection():
    client_module._neo4j_client = None
    assert await check_async_connection()


if __name__ == "__main__":
    logger.info("Starting Neo4j connection tests...")
    
    # Log configuration being used (without exposing password)
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    if not all([uri, username, password]):
        logger.error("❌ Missing Neo4j configuration in environment variables!")
        logger.error(f"   NEO4J_URI: {uri}")
        logger.error(f"   NEO4J_USERNAME: {username}")
        logger.error(f"   NEO4J_PASSWORD: {'***' if password else None}")
        logger.error(f"   NEO4J_DATABASE: {database}")
        sys.exit(1)
    
    logger.info(f"Configuration: URI={uri}, Username={username}, Database={database}")
    logger.info(f"Password length: {len(password) if password else 0} characters")
    
    # Test sync connection
    sync_success = check_sync_connection()
    
    # Reset singleton again before async test
    client_module._neo4j_client = None
    
    # Test async connection
    async_success = asyncio.run(check_async_connection())
    
    if sync_success and async_success:
        logger.info("🎉 All connection tests passed! Neo4j is ready to use.")
    else:
        logger.error("⚠️  Some connection tests failed. Please check your configuration.")

