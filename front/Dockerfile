# Используем базовый образ Node.js
FROM node:22.12-alpine AS build 

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и package-lock.json для установки зависимостей
COPY package*.json ./

# Устанавливаем зависимости с использованием кэша
RUN npm i

# Копируем все файлы проекта в контейнер
COPY . .

# Build your application (if applicable)
RUN npm run build 

# Используем базовый образ Nginx для сервировки
FROM nginx:stable-alpine

# Копируем скомпилированные файлы из стадии сборки
COPY --from=build /app/dist /usr/share/nginx/html 
COPY --from=build /app/nginx.conf /etc/nginx/conf.d/default.conf  

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
