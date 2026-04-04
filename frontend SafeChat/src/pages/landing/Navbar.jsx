import React, { useState } from "react";
import { Menu, X } from "lucide-react";
import { Link } from "react-router-dom";
import Landing from "./Landing";

const Navbar = () => {
  const [open, setOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-white/10 backdrop-blur-md shadow-md w-full">

      <div className="flex justify-between items-center h-22 py-4 w-full px-4">

        {/* Left Corner - Logo */}
        <div className="flex items-center gap-2 md:gap-3 lg:gap-4 ml-4 md:ml-8 lg:ml-16">
          <img src="/logo.png" alt="SafeChat Logo" className="w-6 h-6 md:w-8 md:h-8 lg:w-10 lg:h-10 object-contain"/>
            <span className="text-xl md:text-2xl lg:text-3xl font-bold text-white">
              SafeChat
            </span>
        </div>


        {/* Desktop Menu - Right Corner */}
        <div className="hidden md:flex items-center gap-14 mr-16">
          <a href="#features" className="text-xl font-bold text-white hover:text-orange-400">Features</a>
          <a href="#about" className="text-xl font-bold text-white hover:text-orange-400">About</a>

          <Link to='/user-login'><button className="w-40 md:w-38 lg:w-40 h-10 font-semibold text-base md:text-base lg:text-xl bg-gradient-to-tl from-purple-500 to-slate-700 text-white rounded-md border-2 border-white px-6 py-2 flex items-center justify-center gap-2 hover:from-purple-700 hover:to-slate-900 transition-all">
            Login <span className="text-2xl">&gt;</span>
          </button></Link>
          
        </div>

        {/* Mobile Menu Icon - Extreme Right */}
        <div className="md:hidden">
          <button onClick={() => setOpen(!open)}>
            {open ? <X /> : <Menu />}
          </button>
        </div>

      </div>

      {/* Mobile Dropdown Menu */}
      {open && (
        <div className="md:hidden flex flex-col gap-4 pb-4 px-4">
          <a href="#features" className="hover:text-orange-400">Features</a>
          <a href="#about" className="hover:text-orange-400">About</a>

          <button className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700/30 transition-colors">
            Login
          </button>

        </div>
      )}

    </nav>
  );
};

export default Navbar;
