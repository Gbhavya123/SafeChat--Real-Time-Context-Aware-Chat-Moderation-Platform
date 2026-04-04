function applyPolicy(environment, mlResult) {
    const { is_toxic, severity, suggestion } = mlResult;


    let action = "ALLOW";

    if (!is_toxic) {
        return { action, severity: "LOW", suggestion: null };
    }

    if (environment === "OFFICE") {
        if (severity === "LOW") action = "ALLOW";
        else action = "BLOCK";
    }

    else if (environment === "GAMING") {
        if (severity === "HIGH") action = "SUGGEST"; 
        else action = "ALLOW";
    }

    else {
      
        if (severity === "HIGH") action = "BLOCK";
        else if (severity === "MEDIUM") action = "SUGGEST";
        else action = "ALLOW";
    }

    return {
        action,
        severity,
        suggestion: suggestion || null
    };
}

module.exports = applyPolicy;