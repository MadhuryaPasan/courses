FROM node:25-alpine

# ------------------------------------------------------------------------------------------

WORKDIR /usr/src/app
# The command WORKDIR /usr/src/app (note: the standard Linux path is usually /usr, not /user) is used to define the current working directory for any subsequent commands in your Dockerfile.
# ------------------------------------------------------------------------------------------

ENV NODE_ENV=production
#NODE_ENV is the industry-standard variable used to signal whether the application is running in development or production.

#Less Information Leakage: In development, many frameworks show "Stack Traces" (detailed error reports) directly in the browser if your app crashes. In production mode, these are hidden so attackers can't see your code structure.

# Strict Dependencies: When you run npm install, if NODE_ENV is set to production, npm will skip installing devDependencies (like testing tools or linters), keeping your container smaller and more secure.
# ------------------------------------------------------------------------------------------

COPY package*.json ./
# By copying only package.json and running npm install right after, Docker only re-runs the npm install step if package.json or package-lock.json changes. If you only change your application code (after the RUN npm install layer), Docker uses the cached dependency layer, making builds significantly faster.

# ------------------------------------------------------------------------------------------

RUN --mount=type=cache,target=/usr/src/app/.npm \
  npm set cache /usr/src/app/.npm && \
  npm ci --only=production

# combining two powerful concepts: cache mounting and dependency installation

# `npm ci --only=production` the professional way to install dependencies in a production Docker image. It is faster, more secure, and more reliable than the standard npm install.

# ------------------------------------------------------------------------------------------




# include our code inside the container (copy the src folder)
COPY --chown=node:node ./src .
#(using --chown to keep permissions correct)

# ------------------------------------------------------------------------------------------


EXPOSE 3000
#The EXPOSE 3000 instruction is a form of documentation. It tells the person running the container (and the Docker engine) that the application inside is listening on port 3000.

# ------------------------------------------------------------------------------------------


# to run things on cmd. include the prompt as a list of strings
CMD [ "node", "index.js" ] 
# use case - production $ staging
#The primary reason for using CMD ["node", "index.js"] over CMD ["npm", "run", "dev"] or even CMD ["npm", "start"] in a production Docker image is signal handling.
# When running an application inside a container, you want the main process to be able to receive and handle termination signals (like SIGTERM) from the container orchestrator (Docker, Kubernetes, etc.) so it can shut down cleanly.
