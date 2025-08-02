// src/Chatbot.js
import React from 'react';

const Chatbot = () => {
  return (
    <div style={{ 
      maxWidth: '900px', 
      margin: 'auto', 
      fontFamily: 'Arial, sans-serif', 
      padding: '30px', 
      backgroundColor: 'rgba(255, 255, 255, 0.8)', 
      borderRadius: '10px', 
      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)', 
      position: 'relative',
      overflow: 'hidden', 
      border: '2px solid rgba(0, 0, 0, 0.5)', 
      marginTop: '100px', 
    }}>
      <img 
        src="https://img.freepik.com/free-vector/chatbot-concept-background-realistic-style_23-2147831749.jpg?ga=GA1.1.1095928554.1723716087&semt=ais_hybrid" 
        alt="Chatbot Background" 
        style={{
          position: 'absolute', 
          top: '0', 
          left: '0', 
          width: '100%', 
          height: '100%', 
          objectFit: 'cover', 
          opacity: '0.1' 
        }} 
      />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <img 
          src="https://img.freepik.com/free-vector/floating-robot_78370-3669.jpg?ga=GA1.1.1095928554.1723716087&semt=ais_hybrid" 
          alt="Chatbot Icon" 
          style={{ width: '70px', height: '70px', marginBottom: '15px' }} 
        />
        <h2 style={{ color: '#4CAF50', fontSize: '2em', marginBottom: '30px' }}>Welcome to the Chatbot Application!</h2>
        <p style={{ fontSize: '1.2em', color: '#333' }}>
          Our chatbot is here to assist you with your queries and provide insightful responses 
          tailored to your needs. Whether you require support or simply wish to engage in a conversation, 
          this bot is designed to be your helpful companion!
        </p>
        <p style={{ fontSize: '1.2em', color: '#333' }}>
          To get started, click the link below to access the chatbot application:
        </p>
        <a 
          href="http://localhost:5173/" 
          style={{ 
            textDecoration: 'none', 
            color: '#4CAF50', 
            fontWeight: 'bold', 
            fontSize: '1.5em', 
            display: 'flex', 
            alignItems: 'center' // Align icon and text
          }} 
          target="_blank" 
          rel="noopener noreferrer"
        >
          Open Chatbot App
          <img 
            src="https://img.icons8.com/ios-filled/50/4CAF50/arrow.png" // Arrow icon URL
            alt="Arrow Icon"
            style={{ width: '24px', height: '24px', marginLeft: '10px' }} // Icon styling
          />
        </a>
      </div>
    </div>
  );
};

export default Chatbot;
