/*const jwt = require('jsonwebtoken');
const response = require('../utils/responseHandler');


const authMiddleware = (req,res,next) => {
    const authToken = req.cookies?.auth_token;

    if(!authToken){
        return response(res,401,'autherozation token missing. Please provide token')
    }

    try {
        const decode = jwt.verify(authToken,process.env.JWT_SECRET);
        req.user = decode;
        console.log(req.user);
        next();
    } catch (error) {
        console.error(error);
        return response(res,401,'Invalid or expired Token');
    }
}

module.exports = authMiddleware;*/
const jwt = require('jsonwebtoken');
const response = require('../utils/responseHandler');

const authMiddleware = (req, res, next) => {
    // Support both cookie and Authorization header
    let authToken = req.cookies?.auth_token;

    if (!authToken) {
        const authHeader = req.headers?.authorization;
        if (authHeader && authHeader.startsWith("Bearer ")) {
            authToken = authHeader.split(" ")[1];
        }
    }

    if (!authToken) {
        return response(res, 401, 'autherozation token missing. Please provide token');
    }

    try {
        const decode = jwt.verify(authToken, process.env.JWT_SECRET);
        req.user = decode;
        console.log(req.user);
        next();
    } catch (error) {
        console.error(error);
        return response(res, 401, 'Invalid or expired Token');
    }
};

module.exports = authMiddleware;