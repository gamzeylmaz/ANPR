%% Author: Gamze YILMAZ
%% gamzeyyilmazz@gmail.com
%% Date: 20/12/2018
%
%% Read Input Image
clc;close all;clear all;

im = imread('images/10.jpg');figure, imshow(im);title('Original Image');

%% 1.2-Pre-processing

im = imresize(im, [480 640]);

imgray = rgb2gray(im);%figure, imshow(imgray); title('Gray Image, Size: 480x640');
%imgray=imnoise(imgray,'salt& pepper',0.8);

im2= medfilt2(imgray,[5 5]);%figure, imshow(im2); title('Result of Median Filter');

%% 2.1-Binarization

im3 = edge(im2,'canny',0.3); figure, imshow(im3); title('Canny Algorithm Applied');
% im33 = bwareaopen(im3,70);figure, imshow(im3); title('Cleaned Image After Canny');
[h,w] = size(im3);
output_matrix = zeros(size(im3));

for i = 1:1:h
    for j = 1:1:w-2
        if im3(i,j)== 0
            if im3(i,j+1)== 0
                if im3(i,j+2) == 1
                    output_matrix(i,j) = 1;
                end
            end
        end
        if im3(i,j) == 1
            if im3(i,j+1) == 0
                if im3(i,j+2) == 1
                    output_matrix(i,j) = 1;
                end
            end
        end
    end
end
figure, imshow(output_matrix); title('Output Matrix');
S= sum(output_matrix, 2);   %sum along columns in each row

figure();
subplot(1, 2, 1),imshow(output_matrix);
subplot(1, 2, 2);plot(1: size(S,1), S)
axis([1 size(output_matrix, 1) 0 max(S)]);view(90,90);

% index=find(S==(max(S)));
% cropped=imgray(index(1)-40:index(1)+40,:);
% figure();imshow(cropped);title('cropped image');
%cropped=imgray(i-80 : 1 : h-80,:);figure();imshow(cropped);title('Cropped Image');

[m,i]=max(S);
[~,from] = min(S(1+1:i));
[~,to] = min(S(i:end-1));
cropped=imgray(from:i+to-1,:);figure();imshow(cropped);title('Cropped Image');
%% 3.1 Morphological Transformations

S2 = logical([0 0 1 0 0 ;0 1 1 1 0;1 1 1 1 1;0 1 1 1 0;0 0 1 0 0]);
% S22=strel('disk',1);

GrayEr = imerode(cropped, S2); figure,imshow(GrayEr); title('Eroded Image');
GrayDil = imdilate(GrayEr, S2); figure,imshow(GrayDil); title('Image is Opened');
imcl = imsubtract(GrayDil, GrayEr); figure,imshow(imcl); title('Image is Closed');

gdiff = edge(imcl,'canny', 0.2);figure, imshow(gdiff); title('Canny Algorithm Applied-2');
gdiff2 = bwareaopen(gdiff, 90);figure, imshow(gdiff2); title('Cleaned Image After Canny-2');

gdiff22 = mat2gray(gdiff2);
gdiff2 = conv2(gdiff22,[0 0 0 0 0; 0 1 1 1 0; 0 1 1 1 0; 0 1 1 1 0; 0 0 0 0 0]);
figure, imshow(gdiff2);title('Result of Convulsion');

li= bwlabel(gdiff2, 8);
ifill=imfill(li,8, 'holes');
figure, imshow(ifill);title('Connected Regions Filled');
bm = regionprops(ifill, 'BoundingBox', 'Area');
allAreas = [bm.Area];
[sortedAreas, sortIndexes] = sort(allAreas, 'descend');
figure, imshow(cropped); title('Detected Plate')
hold on
rectangle('Position',[bm(sortIndexes(1)).BoundingBox(1),bm(sortIndexes(1)).BoundingBox(2),bm(sortIndexes(1)).BoundingBox(3),bm(sortIndexes(1)).BoundingBox(4)], 'EdgeColor','r','LineWidth',2 );
hold off


