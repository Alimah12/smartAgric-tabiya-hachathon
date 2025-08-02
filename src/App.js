// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import About from './pages/About';
import DiseasePrediction from './components/DiseasePrediction';
import CropYieldPrediction from './components/CropYieldPrediction';
import CropRecommendation from './components/CropRecommendation';
import Advice from './components/Chatbot';
import EMarket from './components/EMarket'; // Updated import

function App() {
  return (
    <Router>
      <Header />
      <div style={{ display: 'flex', marginTop: '60px' }}> {/* Space for header */}
        <Sidebar />
        <div style={{ flex: 1, padding: '20px', marginLeft: '200px' }}> {/* Adjust padding for sidebar */}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/disease-prediction" element={<DiseasePrediction />} />
            <Route path="/crop-yield-prediction" element={<CropYieldPrediction />} />
            <Route path="/crop-recommendation" element={<CropRecommendation />} />
            <Route path="/chatbot" element={<Advice />} />
            <Route path="/e-market" element={<EMarket />} /> {/* Updated reference */}
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
