// src/pages/About.js
import React from 'react';
import './About.css'; // Assuming you will create a CSS file for styling

// Import images
import aliMalalaImage from '../assets/alimalala.jpg'; // Adjust the path if necessary
import mapesaFadhiliImage from '../assets/fadhil.jpeg'; // Adjust the path if necessary

const About = () => {
  return (
    <div className="about-container">
      <section className="intro-section">
        <h1>About Us</h1>
        <p>
          Welcome to the Smart Agriculture System! Our mission is to empower farmers with advanced tools and resources to enhance agricultural practices, increase productivity, and ensure sustainable farming methods.
        </p>
      </section>
      
      <section className="mission-section">
        <h2>Our Mission</h2>
        <p>
          We are dedicated to bridging the gap between traditional farming methods and modern technology. By providing data-driven insights, we aim to revolutionize the way agriculture is practiced, making it more efficient and sustainable.
        </p>
      </section>

      <section className="features-section">
        <h2>What We Offer</h2>
        <div className="features-list">
          <div className="feature-item">
            <h3>Crop Disease Prediction</h3>
            <p>Utilizing advanced algorithms to identify potential crop diseases early, helping farmers take preventive actions.</p>
          </div>
          <div className="feature-item">
            <h3>Yield Prediction</h3>
            <p>Utilizing machine learning for accurate predictions of crop yields based on various parameters.</p>
          </div>
          <div className="feature-item">
            <h3>Interactive AI Farm Chatbot</h3>
            <p>An AI-driven chatbot that assists farmers with queries in local languages, enhancing communication and support.</p>
          </div>
          <div className="feature-item">
            <h3>Marketplace Connectivity</h3>
            <p>Connecting farmers to a platform where they can sell their products directly to consumers.</p>
          </div>
        </div>
      </section>

      <section className="team-section">
        <h2>Meet Our Team</h2>
        <p>
          Our diverse team consists of agronomists, data scientists, and software engineers, all working together to bring the best agricultural solutions to farmers. We believe in collaboration and continuous improvement to meet the evolving needs of the agricultural sector.
        </p>
        
        <div className="team-members">
          <div className="team-member">
            <img src={aliMalalaImage} alt="Ali Malala" className="team-member-image" />
            <h3>ALI MALALA</h3>
            <p>Lead developer, backend developer</p>
            <p>Ali uses data analytics and machine learning to drive insights that support farmers in making informed decisions. He has a background in agricultural technology.</p>
          </div>
          
          <div className="team-member">
            <img src={mapesaFadhiliImage} alt="Mapesa Fadhili" className="team-member-image" />
            <h3>MAPESA FADHILI</h3>
            <p>Frontend developer</p>
            <p>Mapesa specializes in designing appealing, friendly, and interactive UI.</p>
          </div>
        </div>
      </section>

      <section className="get-in-touch-section">
        <h2>Get In Touch</h2>
        <p>
          We would love to hear from you! Whether you have questions about our system or want to share your feedback, feel free to contact us at <a href="mailto:support@smartagriculture.com">malalaali100@gmail.com</a>.
        </p>
      </section>
    </div>
  );
};

export default About;
