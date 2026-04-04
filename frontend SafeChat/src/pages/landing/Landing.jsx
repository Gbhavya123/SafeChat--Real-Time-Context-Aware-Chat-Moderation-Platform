import React from "react";
import Navbar from "./Navbar";
import { GridScan } from "./GirdScan";
import Footer from "./Footer";
import { Link } from "react-router-dom";

const Landing = () => {
  const avatars = [
    { id: 1, src: "/avtar_1.png", alt: "User 1" },
    { id: 2, src: "/avtar_2.png", alt: "User 2" },
    { id: 3, src: "/avtar_3.png", alt: "User 3" },
    { id: 4, src: "/avtar_4.png", alt: "User 4" },
    { id: 5, src: "/avtar_5.png", alt: "User 5" },
    { id: 6, src: "/avtar_6.png", alt: "User 6" },
    { id: 7, src: "/avtar_7.png", alt: "User 7" },
    // { id: 8, src: "/avtar_8.png", alt: "User 8" },
  ];

  return (
    <div className="relative min-h-screen w-full">

      {/* GridScan Background */}
      <div className="absolute inset-0 -z-10 bg-slate-800">
        <GridScan
          sensitivity={0.55}
          lineThickness={1}
          linesColor="#392e4e"
          gridScale={0.1}
          scanColor="#FF9FFC"
          scanOpacity={0.4}
          enablePost
          bloomIntensity={0.6}
          chromaticAberration={0.002}
          noiseIntensity={0.01}
        />
      </div>

      {/* Main Content */}
      <div className="relative z-10">
        <Navbar />

        {/* Hero Section */}
        <div className="flex flex-col items-center justify-center min-h-[calc(100vh-96px)] px-4 md:px-8 text-center">
          <h1 className="text-2xl md:text-5xl lg:text-4xl font-bold text-white leading-relaxed">
            Make Every Chat Toxic-Free
          </h1>

          <p className="text-white mt-4 text-sm md:text-base lg:text-base max-w-2xl">
            Secure, fast and reliable communication platform.
          </p>

          <Link to="/user-login">
            <button className="mt-6 font-semibold flex items-center justify-center gap-2 px-6 py-2 w-40 md:w-50 lg:w-58 rounded-full border-2 text-xl border-white text-white bg-white/10 hover:bg-purple-300/30 hover:text-white transition-all">
              Login
            </button>
          </Link>
        </div>
      </div>

      {/* About Section */}
      <div className="w-full flex flex-col justify-center items-center text-center px-8 md:px-12 lg:px-28 py-14 bg-gray-900">
        <h2
          id="about"
          className="text-xl md:text-3xl lg:text-4xl font-bold bg-gradient-to-tl from-purple-600 to-pink-700 bg-clip-text text-transparent mb-14"
        >
          About
        </h2>

        <p className="text-base md:text-xl lg:text-xl text-gray-200 max-w-full md:max-w-2xl lg:max-w-3xl px-4 md:px-0 leading-relaxed">
          SafeChat is a user-friendly instant messaging app designed to keep your conversations clean, safe, and enjoyable. 
          Our main goal is to eliminate toxicity from chats, creating a positive environment where users can connect freely 
          without fear of harassment or negativity. With simple navigation, fast messaging, and strong security. 
          Built with care in India, SafeChat combines ease-of-use with responsible communication, making it the perfect
          platform for safe and meaningful interactions.
        </p>
      </div>

      {/* Avatars Section */}
      <div className="w-full flex flex-wrap justify-center items-center gap-6 bg-gray-900 py-4">
        {avatars.map((avatar) => (
          <div key={avatar.id} className="flex flex-col items-center">
            <div className="w-14 h-18 lg:w-32 md:w-27 md:h-30 rounded-full overflow-hidden shadow-lg hover:scale-110 transition-transform">
              <img
                src={avatar.src}
                alt={avatar.alt}
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        ))}
      </div>

      {/* Features Section */}
      <div className="w-full flex flex-col justify-center items-center text-center px-8 md:px-12 lg:px-28 py-20 bg-gray-900">
        <h2
          id="features"
          className="text-xl md:text-3xl lg:text-5xl font-bold bg-gradient-to-tl from-purple-600 to-pink-700 bg-clip-text text-transparent lg:mb-14 text-center"
        >
          Features
        </h2>

        <div className="w-full flex flex-col md:flex-row items-start justify-center md:px-12 lg:px-8 pt-8 gap-10">

          {/* Feature List */}
          <div className="w-full md:w-1/2 text-white text-left">
            <ul className="list-disc list-inside text-base lg:text-xl md:text-xl space-y-2 lg:mt-6">
              <li>Real-time toxicity detection</li>
              <li>Blocks critical harmful messages</li>
              <li>Smart suggestions for better replies</li>
              <li>Secure and private messaging</li>
              <li>Simple and easy-to-use interface</li>
              <li>Promotes positive conversations</li>
            </ul>
          </div>

          {/* Feature Image */}
          <div className="w-full md:w-1/2 flex justify-center items-center">
            <img
              src="/gameGathering.png"
              alt="SafeChat Features"
              className="w-full md:w-[85%] lg:w-[90%] h-52 md:h-[200px] hover:opacity-80 lg:h-[300px] rounded-lg shadow-xl hover:scale-95 object-cover"
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default Landing;
