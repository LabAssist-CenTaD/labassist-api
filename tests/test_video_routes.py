import pytest
import sys
import os

from io import BytesIO

from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
        
def test_upload_video_valid(client):
    data = {'video': (BytesIO(b'video content'), 'video.mp4')}
    response = client.post('/upload', data=data)
    assert response.status_code == 201
    assert response.json == {'message': 'Video uploaded successfully'}
    
    assert 'videos' in client.application.session
    
    assert 'video.mp4' in client.application.session['videos']
    
def test_upload_video_invalid(client):
    data = {'video': (BytesIO(b'video content'), 'video.txt')}
    response = client.post('/upload', data=data)
    assert response.status_code == 400
    assert response.json == {'message': 'Invalid file format. Must be .mp4, .mov, or .avi'}
    
def test_process_video_valid(client):
    data = {'video': (BytesIO(b'video content'), 'video.mp4')}
    client.post('/upload', data=data)
    
    response = client.post('/process_video/video.mp4')
    assert response.status_code == 200
    assert response.json == {'message': 'Video processed successfully'}
    
def test_process_video_invalid(client):
    response = client.post('/process_video/video.mp4')
    assert response.status_code == 404
    assert response.json == {
        'message': 'Video not found or session expired. Please upload the video again',
        'available_videos': []
    }
    
    data = {'video': (BytesIO(b'video content'), 'video.mp4')}
    client.post('/upload', data=data)
    
    response = client.post('/process_video/video.mov')
    assert response.status_code == 404
    assert response.json == {
        'message': 'Video not found or session expired. Please upload the video again',
        'available_videos': ['video.mp4']
    }
    
    response = client.post('/process_video/video.txt')
    assert response.status_code == 404
    assert response.json == {
        'message': 'Video not found or session expired. Please upload the video again',
        'available_videos': ['video.mp4']
    }
