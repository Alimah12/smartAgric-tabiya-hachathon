import React from 'react';
import { Link } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
  const sidebarStyle = {
    position: 'fixed',
    top: '150px', // Increased value for more space
    left: '0',
    width: '200px',
    height: 'calc(100vh - 150px)', // Adjust height accordingly
    backgroundColor: '#f4f4f4',
    padding: '15px',
    boxShadow: '2px 0 5px rgba(0, 0, 0, 0.1)',
  };

  return (
    <div className="sidebar" style={sidebarStyle}>
      <ul>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/about">About Us</Link></li>
        <li><Link to="/disease-prediction">Disease Prediction</Link></li>
        <li><Link to="/crop-yield-prediction">Crop Yield Prediction</Link></li>
        <li><Link to="/crop-recommendation">Crop Recommendation</Link></li>
        <li><Link to="/chatbot">Chatbot</Link></li>
        <li><Link to="/e-market">E-market</Link></li>
      </ul>
    </div>
  );
};

export default Sidebar;
