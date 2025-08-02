// src/AgriculturalEcommerce.js
import React from 'react';
import './EMarket.css'; // Import CSS file for styling

const AgriculturalEcommerce = () => {
  return (
    <div style={{ 
      height: '100vh', 
      backgroundImage: 'url(https://t3.ftcdn.net/jpg/10/25/95/96/360_F_1025959674_a2XBTF8jOCfCCt8EaOf62HMstsUOKVpY.webp)', 
      backgroundSize: 'cover', 
      backgroundPosition: 'center', 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'flex-start', 
      color: 'white', 
      textAlign: 'center', 
      padding: '20px' 
    }}>
      <div style={{ marginTop: '60px' }}></div> {/* Space between header and page content */}
      
      <h1 style={{ 
        fontSize: '3em', 
        marginBottom: '20px', 
        marginTop: '40px', // Space above the header
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.7)', 
        fontWeight: 'bold' 
      }}>
        Welcome to Our Agricultural E-commerce Platform!
      </h1>
      <p style={{ 
        fontSize: '1.5em', 
        marginBottom: '30px', 
        textShadow: '1px 1px 3px rgba(0, 0, 0, 0.7)', 
        fontWeight: 'bold', 
        color: 'white' // Set text color to white
      }}>
        Our platform connects farmers directly with consumers, promoting fresh, organic produce at fair prices. 
        Explore a wide range of agricultural products and support sustainable farming practices.
      </p>

      <a 
        href="http://localhost/Organic-Shopping-Webstore-main/customer_home.php?msg=Login%20Success" 
        style={{ 
          padding: '15px 30px', 
          fontSize: '1.2em', 
          color: 'white', 
          backgroundColor: '#4CAF50', 
          borderRadius: '5px', 
          textDecoration: 'none', 
          boxShadow: '0 4px 10px rgba(0, 0, 0, 0.5)', 
          transition: 'background-color 0.3s' 
        }} 
        onMouseEnter={(e) => (e.target.style.backgroundColor = '#45a049')}
        onMouseLeave={(e) => (e.target.style.backgroundColor = '#4CAF50')}
      >
        Go to Organic Shopping
      </a>

      <div style={{ marginTop: '60px' }}></div> {/* Increased space between page content and sidebar */}
    </div>
  );
};

export default AgriculturalEcommerce;
