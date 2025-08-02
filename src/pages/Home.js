import React from 'react';
import Header from '../components/Header'; // Updated path for Header component
import './Home.css';
import { FaSeedling, FaLeaf, FaTools, FaInfoCircle } from 'react-icons/fa';

const features = [
  {
    icon: <FaSeedling />,
    title: "Yield Prediction",
    description: "Get accurate predictions on crop yield based on various parameters.",
  },
  {
    icon: <FaLeaf />,
    title: "Disease Prediction",
    description: "Identify crop diseases from images and receive treatment suggestions.",
  },
  {
    icon: <FaTools />,
    title: "Crop Recommendation",
    description: "Receive recommendations on the best crops to grow based on conditions.",
  },
  {
    icon: <FaInfoCircle />,
    title: "Interactive AI Farm Chatbot",
    description: "Connect with our AI chatbot for personalized farming advice, including local language translation.",
  },
];

const Home = () => {
  const backgroundImageStyle = {
    backgroundImage: `url(${process.env.PUBLIC_URL + '/background.png'})`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    minHeight: '30vh',
    padding: '4px 4px',
  };

  return (
    <>
      <Header /> {/* Fixed Header */}
      <div style={{ marginTop: '40px' }}> {/* Increased space between header and page */}
        <div
          className="home-container"
          style={{ ...backgroundImageStyle, marginTop: '20px', marginLeft: '30px' }} // Maintained margin-left for space from sidebar
        >
          <h1 className="welcome-title">Welcome to the Smart Agriculture System</h1>
          <p className="intro-text">
            Our mission is to empower farmers with advanced tools and resources to enhance agricultural practices and ensure sustainable farming methods.
          </p>

          <section className="what-we-offer" style={{ marginBottom: '20px' }}> {/* Space below what we offer section */}
            <h2>What We Offer</h2>
            <p>
              We provide innovative solutions, including:
              <ul>
                <li>Accurate yield predictions</li>
                <li>Crop disease identification</li>
                <li>Personalized crop recommendations</li>
                <li>An interactive AI chatbot to assist farmers</li>
              </ul>
            </p>
          </section>

          {/* Features Section */}
          <div className="features-container">
            {features.map((feature, index) => (
              <div className="feature" key={index}>
                <div className="feature-icon">{feature.icon}</div>
                <h2>{feature.title}</h2>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>

          <div className="footer-note">
            <p>Explore our features by clicking the buttons above!</p>
          </div>
        </div>
      </div>
    </>
  );
};

export default Home;
