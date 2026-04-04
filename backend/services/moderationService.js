const axios = require("axios");

const moderationEngine = async (text) => {
  try {
    const response = await axios.post(
      "https://vineet88-context-aware-safety-ml-api.hf.space/api/v1/moderate",
      { text: text }
    );

    return response.data;
  } catch (error) {
    console.error("Moderation error:", error.message);
    throw new Error("ML service failed");
  }
};

module.exports = moderationEngine;
