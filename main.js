// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const documentFile = document.getElementById('documentFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const statusElement = document.getElementById('status');
    
    // Load knowledge base status
    fetchKnowledgeBaseStatus();
    
    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    uploadBtn.addEventListener('click', uploadDocument);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Initial message
    addMessage('bot', 'Hello! I am your document RAG chatbot. You can ask me questions about the documents in my knowledge base or upload new documents for me to learn from.');
    
    // Function to send message
    function sendMessage() {
        const query = userInput.value.trim();
        
        if (query === '') return;
        
        // Add user message to chat
        addMessage('user', query);
        userInput.value = '';
        
        // Show loading indicator
        const loadingId = showLoading();
        
        // Send query to backend
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            hideLoading(loadingId);
            
            // Add bot response to chat
            addMessage('bot', data.answer, data.sources);
        })
        .catch(error => {
            // Remove loading indicator
            hideLoading(loadingId);
            
            console.error('Error:', error);
            addMessage('bot', 'Sorry, there was an error processing your request.');
        });
    }
    
    // Function to upload document
    function uploadDocument() {
        const file = documentFile.files[0];
        
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            uploadStatus.style.backgroundColor = '#ffcccc';
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        uploadStatus.textContent = 'Uploading...';
        uploadStatus.style.backgroundColor = '#ffffcc';
        
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                uploadStatus.textContent = data.message;
                uploadStatus.style.backgroundColor = '#ccffcc';
                documentFile.value = ''; // Clear the file input
                
                // Refresh knowledge base status
                fetchKnowledgeBaseStatus();
                
                // Add a message to the chat about the new document
                addMessage('bot', `I've processed "${file.name}" and added it to my knowledge base. You can now ask me questions about it.`);
            } else {
                uploadStatus.textContent = data.message;
                uploadStatus.style.backgroundColor = '#ffcccc';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            uploadStatus.textContent = 'Error uploading file.';
            uploadStatus.style.backgroundColor = '#ffcccc';
        });
    }
    
    // Function to add message to chat
    function addMessage(role, text, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const sender = role === 'user' ? 'You' : 'Bot';
        messageDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'source';
            sourceDiv.innerHTML = `Sources: ${sources.join(', ')}`;
            messageDiv.appendChild(sourceDiv);
        }
        
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    // Function to show loading indicator
    function showLoading() {
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.className = 'message bot';
        loadingDiv.innerHTML = '<strong>Bot:</strong> <em>Thinking...</em>';
        
        chatbox.appendChild(loadingDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
        
        return loadingId;
    }
    
    // Function to hide loading indicator
    function hideLoading(loadingId) {
        const loadingDiv = document.getElementById(loadingId);
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }
    
    // Function to fetch knowledge base status
    function fetchKnowledgeBaseStatus() {
        fetch('/api/prebuilt-status')
            .then(response => response.json())
            .then(data => {
                statusElement.innerHTML = 
                    `Knowledge Base Status: ${data.total_chunks} chunks from ${data.document_count} documents`;
            })
            .catch(error => {
                console.error('Error fetching status:', error);
                statusElement.innerHTML = 'Error loading knowledge base status';
            });
    }
});