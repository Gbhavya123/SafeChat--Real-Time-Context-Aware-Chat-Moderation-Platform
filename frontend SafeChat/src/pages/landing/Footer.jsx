import React from "react";

const Footer = () => {
  return (
    <footer className="text-white py-8 px-4 relative bg-purple-800">
      <div className="container mx-auto flex items-center justify-center md:justify-between">

        {/* Centered Copyright */}
        <div className="text-base md:absolute md:left-1/2 md:-translate-x-1/2 text-center">
          © {new Date().getFullYear()} SafeChat. All rights reserved.
        </div>

      </div>
    </footer>
  );
};

export default Footer;
