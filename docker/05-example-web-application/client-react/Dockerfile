FROM node:25-bookworm AS build

WORKDIR /usr/src/app

COPY package*.json ./


RUN --mount=type=cache,target=/usr/src/app/.npm \
npm set cache /usr/src/app/.npm && \
npm ci 

COPY . .

RUN npm run build


###
FROM nginxinc/nginx-unprivileged:1.29-alpine-perl

COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build usr/src/app/dist/ /usr/share/nginx/html

EXPOSE 8080