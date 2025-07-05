import React, { useEffect, useRef, useState } from "react";
import { chatApi } from "../services/api";
import { useTenant } from "./TenantContext";

function Chat() {
  const { tenant } = useTenant();
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [sender, setSender] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);

  // Load messages on component mount
  useEffect(() => {
    loadMessages();
    // Set up polling to refresh messages every 3 seconds
    const interval = setInterval(loadMessages, 3000);
    return () => clearInterval(interval);
  }, []);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadMessages = async () => {
    try {
      const data = await chatApi.getMessages();
      setMessages(data);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!newMessage.trim() || !sender.trim()) {
      setError("Please enter both sender name and message");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      await chatApi.sendMessage({
        message: newMessage.trim(),
        sender: sender.trim(),
      });

      setNewMessage("");
      await loadMessages(); // Refresh messages immediately after sending
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  // Clear error after 3 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError("");
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>💬 Chat Room</h2>
        <div className="tenant-info">
          <span className="tenant-label">Company:</span>
          <span className="tenant-name">{tenant.name}</span>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="no-messages">
            No messages yet. Start the conversation!
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className="message">
              <div className="message-header">
                <span className="sender">{message.sender}</span>
                <span className="timestamp">
                  {formatTimestamp(message.created_at)}
                </span>
              </div>
              <div className="message-content">{message.message}</div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="chat-input-form">
        <div className="input-group">
          <input
            type="text"
            value={sender}
            onChange={(e) => setSender(e.target.value)}
            placeholder="Your name"
            className="sender-input"
            disabled={isLoading}
          />
          <input
            type="text"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            placeholder="Type your message..."
            className="message-input"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !newMessage.trim() || !sender.trim()}
            className="send-button"
          >
            {isLoading ? "Sending..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}

export default Chat;
