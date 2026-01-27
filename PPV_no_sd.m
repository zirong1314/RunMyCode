%%  多尺度融合
clear all;
clc;
tic
baseFileName = { 'NV-1', 'NV-2', 'NV-3', 'NV-4', 'NV-5', 'NV-6', 'NV-7'};
fileType = '.txt'; 
numFiles = 10;% 共36组数据
for t = 1:numFiles
% 构建完整的文件名
filename = [char(baseFileName(t)),fileType];
M = readmatrix(filename);
input=M(2189:5900,2);
%% 求峰值谱
j=1;
dt=0.0001;% 一个格9.7mm
v=350/3.6;
max_window=240;
min_window=120;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%非常重要的参数
step=5;% 五个格50mm 0.05m 5cm
interval=5;
for i=min_window:interval:max_window
    [y, x]=PPV_method(input, i, max_window, step, dt, v);% 输入 窗长 步长 采样间隔 车速 
    len=length(y);
    P(1:len,j)=y;%每个尺度下的值
    dist(1:len,j)=x;%每个尺度下的值对应的中心位置点
    j=j+1;
end
%% 去除噪音
%设置检测宽度
% W=80;
% numk=length(P(:,1))-W+1;
% for k=1:1:numk
%     win=P((W/2+(k-1))-W/2+1:(W/2+(k-1))+W/2,:);
%     PC(k,:)=sum(win);
% end
% 绘图
figure
x_axis = dist(:,1);                          % 距离（m）
y_axis = 1:length(min_window:step:max_window);     % 窗口尺度
h = imagesc(x_axis, y_axis, P');
axis xy
colorbar;
caxis([-30, 80]); % 颜色范围设置
xlabel('Distance(m)','fontsize',20,'FontName','Times New Roman','Fontweight','bold');
ylabel('Window scale','fontsize',20,'FontName','Times New Roman','Fontweight','bold');
end
toc
