clear all; close all; clc

dt=0.01;
t=0:dt:50; 
x0=[0.1 5];
mu=1.2;

[t,y]=ode45('rhs_dyn',t,x0,[],mu);

plot(t,y(:,1),t,y(:,2),'Linewidth',[2])

x1=y(:,1);
x2=y(:,2);


%%
H1=[x1(1:4000).'
   x2(1:4000).'
   x1(2:4001).'
   x2(2:4001).'
   x1(3:4002).'
   x2(3:4002).'
   x1(4:4003).'
   x2(4:4003).'
   x1(5:4004).'
   x2(5:4004).'
   x1(6:4005).'
   x2(6:4005).'];

H2=[x1(1:4000).'
   x2(1:4000).'
   x1(2:4001).'
   x2(2:4001).'
   x1(3:4002).'
   x2(3:4002).'
   x1(4:4003).'
   x2(4:4003).'
   x1(5:4004).'
   x2(5:4004).'
   x1(6:4005).'
   x2(6:4005).'
   x1(7:4006).'
   x2(7:4006).'
   x1(8:4007).'
   x2(8:4007).'
   x1(9:4008).'
   x2(9:4008).'
   x1(10:4009).'
   x2(10:4009).'];

figure(3)
[u,s,v]=svd(H1,'econ');
subplot(2,1,1), plot(diag(s)/(sum(diag(s))),'ro','Linewidth',[3])
figure(4), subplot(2,1,1), plot(u(:,1:3),'Linewidth',[2])
subplot(2,1,2), plot(v(:,1:3),'Linewidth',[2])
figure(77), plot(v(:,1:2)); hold on

figure(3)
[u,s,v]=svd(H2,'econ');
subplot(2,1,2), plot(diag(s)/(sum(diag(s))),'ro','Linewidth',[3])
figure(5), subplot(2,1,1), plot(u(:,1:3),'Linewidth',[2])
subplot(2,1,2), plot(v(:,1:3),'Linewidth',[2])
figure(77), plot(v(:,1:2)); hold on




%%

H3=[]
for j=1:900
  H3=[H3; y(j:4000+j,:).']; 
end   

figure(6)
[u,s,v]=svd(H3,'econ');
subplot(2,1,2), plot(diag(s)/(sum(diag(s))),'ro','Linewidth',[3])
figure(7), subplot(2,1,1), plot(u(:,1:3),'Linewidth',[2])
subplot(2,1,2), plot(v(:,1:3),'Linewidth',[2])


