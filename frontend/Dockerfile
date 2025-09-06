# InvestIQ Platform Frontend - Next.js Application
# Multi-stage Docker build for optimized production deployment

# 开发阶段
FROM node:18-alpine AS development

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY frontend/package*.json ./
RUN npm ci --only=development

# 复制源代码
COPY frontend/ .

# 暴露开发服务器端口
EXPOSE 3000

# 开发命令
CMD ["npm", "run", "dev"]

# 构建阶段
FROM node:18-alpine AS builder

WORKDIR /app

# 安装依赖
COPY frontend/package*.json ./
RUN npm ci --only=production && npm cache clean --force

# 复制源代码
COPY frontend/ .

# 构建应用
RUN npm run build

# 生产阶段
FROM node:18-alpine AS production

WORKDIR /app

# 创建非root用户
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# 只复制必要的文件
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json

# 复制构建产物
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# 设置用户
USER nextjs

# 暴露端口
EXPOSE 3000

# 设置环境变量
ENV PORT 3000
ENV NODE_ENV production

# 启动应用
CMD ["node", "server.js"]

# Nginx代理阶段（可选，用于生产环境）
FROM nginx:alpine AS nginx

# 复制构建的静态文件
COPY --from=builder /app/out /usr/share/nginx/html

# 复制nginx配置
COPY frontend/nginx.conf /etc/nginx/nginx.conf

# 暴露端口
EXPOSE 80

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]