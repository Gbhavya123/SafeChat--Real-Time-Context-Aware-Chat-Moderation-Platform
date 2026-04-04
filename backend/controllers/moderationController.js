const moderationEngine = require("../services/moderationService");

exports.analyzeMessage = async (req, res) => {
  try {
    const { message } = req.body;

    if (!message || !message.trim()) {
      return res.status(400).json({
        level: "Easy",
        is_flagged: false,
        suggested_message: ""
      });
    }

    const result = await moderationEngine(message);

    // Normalize severity (numeric-safe)
    const sev = Number(
      result.severity_level ??
      result.severity ??
      result.level ??
      1
    );

    let level = "Easy";
    if (sev >= 3) level = "Critical";
    else if (sev === 2) level = "Medium";
    // ✅ CLEAN suggestion (NOT ML-generated)
    const suggested_message = getCleanSuggestion(level, message);

    return res.status(200).json({
      level,
      is_flagged: Boolean(result.is_flagged),
      suggested_message
    });

  } catch (error) {
    console.error("Moderation error:", error);
    return res.status(500).json({
      level: "Easy",
      is_flagged: false,
      suggested_message: ""
    });
  }
};
function getCleanSuggestion(level, originalMessage) {
  switch (level) {
      case "Critical":
      return "Please rewrite this message using respectful and non-offensive language.";
    case "Medium":
      return "Consider softening your tone to keep the conversation respectful.";
    default:
      return originalMessage;
  }
}

