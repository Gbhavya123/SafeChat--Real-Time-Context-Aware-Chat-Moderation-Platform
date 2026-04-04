import React, { useState, useEffect } from "react";
import useLayoutStore from "../../store/LayoutStore"; // import store
import useThemeStore from "../../store/themeStore";

const Setting = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const { theme: globalTheme, setTheme: setGlobalTheme } = useThemeStore();
  const [theme, setTheme] = useState(globalTheme);
  const [notifications, setNotifications] = useState(true);

  // ✅ use Zustand for read receipts
  const readReceipts = useLayoutStore((state) => state.readReceipts);
  const setReadReceipts = useLayoutStore((state) => state.setReadReceipts);

  // Load saved settings
  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("chatSettings"));
    if (saved) {
      setUsername(saved.username);
      setEmail(saved.email);
      setTheme(saved.theme);
      setNotifications(saved.notifications);
      setReadReceipts(saved.readReceipts); // update store
    }
  }, [setReadReceipts]);

  const handleSave = () => {
  const data = {
    username,
    // email,
    theme,
    notifications,
    readReceipts,
  };
  localStorage.setItem("chatSettings", JSON.stringify(data));

  // ✅ Update global Zustand theme
  setGlobalTheme(theme);

  alert("Settings Saved!");
};

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 p-4 sm:p-8">
      <div className="max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow-md p-4 sm:p-6">
        <h2 className="text-2xl sm:text-3xl font-bold text-center mb-6 text-gray-800 dark:text-white">
          Chat Settings
        </h2>

        {/* Profile Section */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">
            Profile Settings
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <input
              className="p-3 border rounded-lg w-full focus:outline-none"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />

            <input
              className="p-3 border rounded-lg w-full focus:outline-none"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
        </div>

        {/* Appearance */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">
            Appearance
          </h3>

          <select value={theme}
            onChange={(e) => {
              setTheme(e.target.value);
              setGlobalTheme(e.target.value); // immediate update
            }}
            className="p-3 border rounded-lg w-full"
          >
            <option value="light">Light Mode</option>
            <option value="dark">Dark Mode</option>
        </select>

        </div>

        {/* Notifications */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">
            Notifications
          </h3>

          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={notifications}
              onChange={() => setNotifications(!notifications)}
              className="w-5 h-5"
            />
            <span className="text-gray-700 dark:text-gray-300">
              Enable Notifications
            </span>
          </div>
        </div>

        {/* Privacy */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">
            Privacy
          </h3>

          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={readReceipts}
              onChange={(e) => setReadReceipts(e.target.checked)}
              className="w-5 h-5"
            />
            <span className="text-gray-700 dark:text-gray-300">
              Show Read Receipts
            </span>
          </div>
        </div>

        {/* Save Button */}
        <button
          onClick={handleSave}
          className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition"
        >
          Save Settings
        </button>
      </div>
    </div>
  );
};

export default Setting;
