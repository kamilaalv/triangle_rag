from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Chat(Base):
    __tablename__ = "chats"
    chat_id = Column(String, primary_key=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    chat_id = Column(String)
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Function to initialize the database and create tables
def init_db():
    db_path = 'data/chat.db'
    
    # If the directory doesn't exist, create it
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))
    
    # Create SQLite engine
    engine = create_engine(f'sqlite:///{db_path}', echo=True)
    
    # Create all tables if they do not exist
    Base.metadata.create_all(engine)
    
    # Return a sessionmaker bound to the engine
    return sessionmaker(bind=engine)

# Create a session class using the init_db function
SessionLocal = init_db()

# Dependency to get the database session
def get_session():
    session = SessionLocal()  # Create a new session
    try:
        yield session  # Provide the session to the caller
    except Exception as e:
        session.rollback()  # Rollback in case of error
        print(f"Error in transaction: {e}")
        raise
    finally:
        session.close()  # Always close the session after the operation
