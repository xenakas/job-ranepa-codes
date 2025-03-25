function MainFunction = Main_Function_Tool(index_t_s, index_t_r, index_indicators, index_years, index_results, index_output, folder_name);
% This function is run after taking the selections in the MAIN_TOOL.mlapp.
% This function calculates the scope 3 impacts of target-sector regions 
% for each indicator and year, and from different perspectives to track 
% these impacts upstream and downstream the global value chain.
% Results are saved as textfiles in the created folder.


%% derive the index of target-sector-regions (t) and non-target sector-regions (o)
index_all_s = [1:163];
index_all_r = [1:49];

index_o_s = setdiff(index_all_s, index_t_s);
index_o_r = setdiff(index_all_r, index_t_r);

z = 0;
for n = index_t_r;
for i = index_t_s;
q = i + (163 * (n-1)) ;
z= z+1;
index_t(z)= q;
end
end
n_t = length(index_t);

index_all = [1:7987];
index_o = setdiff(index_all, index_t);
n_o = length(index_o);

n_t_s = length(index_t_s);
n_t_r = length(index_t_r);
n_o_s = length(index_o_s);

% further index
no_indicators = length(index_indicators);   
no_years = length(index_years);



%% some index for data compilation later
for n = 1:49
    i = 1 + (n-1)*163;
    j = i + 162;
    index_matrix_P(n,:) = i:j; % index for Production Perspective
end

for n = 1:n_t_r
    i = 1 + (n-1)*n_t_s;
    j = i + (n_t_s-1);
    index_matrix_T(n,:) = i:j; % index for Target Perspective
end

for n = 1:49
    i = 1 + (n-1)*7;
    j = i + 6;
    index_matrix_FD(n,:) = i:j; % index for Final Demand Perpsective
end


%% load labels and create new names and labels
datapath = ['Input_Data/Labels/'];
load([datapath 'Labels_Regions_all_Tool']);
load([datapath 'Labels_Sectors_all_Tool']);
load([datapath 'Labels_Production_Tool']);
load([datapath 'Labels_FinalDemand_Tool']);
load([datapath 'Labels_FinalDemandCategories']);
load([datapath 'Labels_Indicators']);

Labels_Target_Sectors_Tool = Labels_Sectors_all_Tool(index_t_s); % names of the selected target sectors
Labels_Target_Regions_Tool = Labels_Regions_all_Tool(index_t_r); % names of the selected target regions
Labels_Target_Tool = Labels_Production_Tool(index_t); % Labels of the Target stage
Labels_FinalSupply_Tool(1,1) = {'Direct final demand for target-sector-region-outputs'};
Labels_FinalSupply_Tool(2:(n_o+1),1) = Labels_Production_Tool(index_o); % Labels of the Final Supply stage

% Save your settings in the folder "Your_Settings"
mkdir(['' folder_name '/Your_Settings/']); 
datapath = ['' folder_name '/Your_Settings/'];
writetable(table(Labels_Target_Sectors_Tool),[datapath 'Your_selected_Target_Sectors.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
writetable(table(Labels_Target_Regions_Tool),[datapath 'Your_selected_Target_Regions.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
writetable(table(Labels_Indicators(index_indicators)),[datapath 'Your_selected_Indicators.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
writetable(table(index_years'),[datapath 'Your_selected_Years.txt'], 'WriteVariableNames',false, 'delimiter', '\t');


%% Procedure for each year
k_time = 0;
for year = index_years;  
k_time = k_time + 1; % increase the index for the timelines

% Read in coefficient matrix A, final demand matrix FD, environmental extensions Q,
% household emissions Q_hh
datapath = ['Input_Data/IOT_' num2str(year) '_ixi/'];
A = dlmread([datapath 'A.txt'],'\t',3,2); % A = coefficient matrix with 7987 rows x 7987 columns (7987 = 163 sectors x 49 regions)
Y = dlmread([datapath 'Y.txt'],'\t',3,2); % Y = final demand with 7987 rows x 343 columns (343 = 49 regions x 7 final demand categories)
datapath = ['Input_Data/IOT_' num2str(year) '_ixi/satellite/'];
Q = dlmread([datapath 'F.txt'],'\t',2,1); % Q = environmental and social extensions (rows) for 7987 sector-region combinations (columns)
if sum(ismember(index_output,2)) > 0; % only if global shares are calculated
Q_hh = dlmread([datapath 'F_hh.txt'],'\t',2,1); % Q_hh = environmental and social extensions (rows) for households (columns = 49 regions x 7 final demand categories)
end

% Reorder data for Tool (other sequence than original data)
datapath = ['Input_Data/Conversion/'];
load([datapath 'Regions_No_Tool']);
load([datapath 'Sectors_No_Tool']);
load([datapath 'Production_No_Tool']);
load([datapath 'FinalDemand_No_Tool']);
A = A(Production_No_Tool,Production_No_Tool);
Y = Y(Production_No_Tool,FinalDemand_No_Tool);
Q = Q(:,Production_No_Tool);
if sum(ismember(index_output,2)) > 0; % only if global shares are calculated
Q_hh = Q_hh(:,FinalDemand_No_Tool);
end

% Calculate Leontief Inverse (L)
I = eye(size(A));
L = inv(I-A);

% Derive Total output of each sector-region combination to satisfy the global
% final demand (TotalOut)
X = L*Y;
TotalOut =sum(X,2);

%% Derive the total output without double counting by applying the method of Dente et al. (2018, 2019)

% Derive the total output without double counting as a vector (x_t_wdc)
L_oo_dash = inv(I(index_o,index_o)-A(index_o,index_o));
Y_global = sum(Y,2);
x_t_wdc(:,1) = Y_global(index_t,1) + A(index_t,index_o) * L_oo_dash * Y_global(index_o,1); 

% Derive the total output without double counting as matrices
if sum(ismember(index_results,[1:39])) > 0; % only necessary when different perspectives are calculated
X_t_wdc_C(:,1:343) = Y(index_t,:) + A(index_t,index_o) * L_oo_dash * Y(index_o,:); %link between target (rows) and final demand (column)
X_t_wdc_O(:,1) = Y_global(index_t);
X_t_wdc_O(:,2:(n_o+1)) = A(index_t,index_o) * L_oo_dash * diag(Y_global(index_o)); %link between target (rows) and final supply(column)
end





%% Derive all environmental and socio-economic indicators of the global economy (Ind)
% *Indicators covered in the Global Resource Outlook (2019) and Cabernard et al. (2019)

% 1. Material footprint (kt)*
Ind(1,:) = sum(Q([471:687],:));

% 2. Unused material footprint (kt)
Ind(2,:) = sum(Q([688:910],:));

% 3. Climate change impacts: total (kg CO2-eq)*
CF_CH4 = 28;
CF_N2O = 265;
Ind(3,:) = sum(Q([24,93,94,428,438,439],:)) + sum(Q([25,68:75,427,436],:)) .* CF_CH4 + sum(Q([26,430],:)) .* CF_N2O + Q(425,:) + Q(426,:);

% 4. CO2-related climate change impacts (kg CO2-eq)
Ind(4,:) = sum(Q([24,93,94,428,439],:));

% 5. CH4-related climate change impacts (kg CO2-eq)
CF_CH4 = 28;
Ind(5,:) = sum(Q([25,68:75,427,436],:)) .* CF_CH4;

% 6. N2O-related climate change impacts (kg CO2-eq)
CF_N2O = 265;
Ind(6,:) = sum(Q([26,430],:)) .* CF_N2O;

% 7. HFC-related climate change impacts (kg CO2-eq)
Ind(7,:) = Q([425],:);

% 8. PFC-related climate change impacts (kg CO2-eq)
Ind(8,:) = Q([426],:);

% 9. Particulate-matter rel. health impacts: total (DALYs)*
% define sector index
mining = [1:19,32:35,42:49,64:66];
processing = [20:31,36:41,50:63,67:75];
other = [76:163];
% CFs for PM25
CF_PM25_mining = 0.0002320757; 
CF_PM25_processing = 0.0011171740; 
CF_PM25_other = 0.0021326226; 
for i = mining;
for n = 0:48;
    j = i + 163*n;
CF_PM25(j) = CF_PM25_mining ; 
end
end
for i = processing;
for n = 0:48;
    j = i + 163*n;
CF_PM25(j) = CF_PM25_processing ;
end
end
for i = other;
for n = 0:48;
    j = i + 163*n;
CF_PM25(j) = CF_PM25_other ;
end
end
% CFs for SOx
CF_SOx_mining = 0.0000190000;
CF_SOx_processing = 0.0000845000;
CF_SOx_other = 0.0001500000;
for i = mining;
for n = 0:48;
    j = i + 163*n;
CF_SOx(j) = CF_SOx_mining ;
end
end
for i = processing;
for n = 0:48;
    j = i + 163*n;
CF_SOx(j) = CF_SOx_processing ;
end
end
for i = other;
for n = 0:48;
    j = i + 163*n;
CF_SOx(j) = CF_SOx_other ;
end
end
% CFs for NOx
CF_NOx_mining = 0.0000040000;
CF_NOx_processing = 0.0000175000;
CF_NOx_other = 0.0000310000;
for i = mining;
for n = 0:48;
    j = i + 163*n;
CF_NOx(j) = CF_NOx_mining ;
end
end
for i = processing;
for n = 0:48;
    j = i + 163*n;
CF_NOx(j) = CF_NOx_processing ;
end
end
for i = other;
for n = 0:48;
    j = i + 163*n;
CF_NOx(j) = CF_NOx_other ;
end
end
% CFs for NH3
CF_NH3_mining = 0.0000400000;
CF_NH3_processing = 0.0001500000;
CF_NH3_other = 0.0002600000;
for i = mining;
for n = 0:48;
    j = i + 163*n;
CF_NH3(j) = CF_NH3_mining ;
end
end
for i = processing;
for n = 0:48;
    j = i + 163*n;
CF_NH3(j) = CF_NH3_processing ;
end
end
for i = other;
for n = 0:48;
    j = i + 163*n;
CF_NH3(j) = CF_NH3_other ;
end
end
Ind(9,:) = sum(Q([40,283:328,445],:)) .* CF_PM25 + sum(Q([27,343:361,446],:)) .* CF_SOx + sum(Q([28,189:210,432,443],:)) .* CF_NOx + sum(Q([29,141,431,442],:)) .* CF_NH3;

% 10. PM2.5-related health impacts (DALYs)
Ind(10,:) = sum(Q([40,283:328,445],:)) .* CF_PM25;

% 11. SOx-related health impacts (DALYs)
Ind(11,:) = sum(Q([27,343:361,446],:)) .* CF_SOx;

% 12. NOx-related health impacts (DALYs)
Ind(12,:) = sum(Q([28,189:210,432,443],:)) .* CF_NOx;

% 13. NH3-related health impacts (DALYs)
Ind(13,:) = sum(Q([29,141,431,442],:)) .* CF_NH3 ;


% 14. Water Stress (Mio. m3 H2O-eq)*
% Calculate Water Stress (WS) in Mio m3 H2O equivalents
% Water consumption reported by the satellite matrix of EXIOBASE 3.4 was weighted 
% with sector-region specific impact factors to derive water stress.
datapath = ['Input_Data/CF/'];
CF_water = dlmread([datapath 'CF_water.txt'],'\t',2,2); % impact factors to derive water stress
%Define CFs for region-sector comb.
index_CF_water = [1,2,3,4,5,6,7,8,repmat(9,1,11),repmat(10,1,144)]; % assign the impact factors to the respective sectors
n = 0;
for i = 1:49
for j = index_CF_water   
n = n+1;    
CF_water_rs(n) = CF_water(i,j);
end
end
% extensions of the economy
WS = sum(Q([924:1026],:));
Ind(14,:)  = CF_water_rs .* WS;
clear WS

% 15. Blue water consumption (Mio. m3 H2O)
Ind(15,:) = sum(Q([924:1026],:));

% 16. Green water consumption (Mio. m3 H2O)
Ind(16,:) = sum(Q([911:923],:));

% 17. Land-use related biodiversity loss (global PDF)*
% Calculate Land-use related Biodiversity Loss (LU) in global PDF
% Read in data to derive CFs for land-use related-biodiversity loss
datapath = ['Input_Data/CF/'];
CF_landuse = dlmread([datapath 'CF_landuse.txt'],'\t',2,2);
CF_landuse(:, 11) = 0; % to set "other land use" = 0
Q_2007 = dlmread([datapath 'F_2007.txt'],'\t',2,1);
Q_2007 = Q_2007(:,Production_No_Tool);

% Derive CFs for the economy
% Extensions of the year 2007 (necessary to derive CFs)
Ext_2007(1,:) = Q_2007(447,:); %cereal grains nec
Ext_2007(2,:) = Q_2007(448,:); %crops nec
Ext_2007(3,:) = sum(Q_2007(449:453,:)); % sum of fodder crops
Ext_2007(4,:) = Q_2007(454,:); % oil seeds
Ext_2007(5,:) = Q_2007(455,:); % paddy rice
Ext_2007(6,:) = Q_2007(456,:); % plant-based fibers
Ext_2007(7,:) = Q_2007(457,:); % sugar cane & sugar beet
Ext_2007(8,:) = Q_2007(458,:); % vegetables, fruits, nuts
Ext_2007(9,:) = Q_2007(459,:); % wheat
Ext_2007(10,:) = Q_2007(460,:); % forestry
Ext_2007(11,:) = Q_2007(461,:); % other land use (=0)
Ext_2007(12,:) = sum(Q_2007(462:464,:)); % permanent pastures
Ext_2007(13,:) = Q_2007(465,:); % infrastructure land
Ext_2007(14,:) = Q_2007(466,:); % forest area marginal use

% Extensions of the chosen year
Ext_year(1,:) = Q(447,:); % same description as for 2007
Ext_year(2,:) = Q(448,:);
Ext_year(3,:) = sum(Q(449:453,:));
Ext_year(4,:) = Q(454,:);
Ext_year(5,:) = Q(455,:);
Ext_year(6,:) = Q(456,:);
Ext_year(7,:) = Q(457,:);
Ext_year(8,:) = Q(458,:);
Ext_year(9,:) = Q(459,:);
Ext_year(10,:) = Q(460,:);
Ext_year(11,:) = Q(461,:);
Ext_year(12,:) = sum(Q(462:464,:));
Ext_year(13,:) = Q(465,:);
Ext_year(14,:) = Q(466,:);

for q = 1:14;
for n = 0:48;
i = 1 + 163*n;
j = i + 162;
z = n+1;
CF_landuse_e(q,i:j) = CF_landuse(z,q);
end
end

for q = 1:9
for n = 0:48;
    i = 1 + 163*n;
    j = i + 162;
for z =  i:j;  
    if Ext_2007(q,z)>0
        CF_landuse_final(q,z)= CF_landuse_e(q,z) / sum(sum(Ext_2007(q,i:j)));
    else
        CF_landuse_final(q,z)= 0;  
    end
end
end
end

CF_landuse_final(10:14,:) = CF_landuse_e(10:14,:)*10^6;
Ind(17,:) =  sum(CF_landuse_final .* Ext_year);

% 18. Land-use area - total (km2)
Ind(18,:) = sum(Q([447:466],:));

% 19. Cropland (km2)
Ind(19,:) = sum(Q([447:459],:));

% 20. Pemanent pastures (km2)
Ind(20,:) = sum(Q([462:464],:));

% 21. Forest area: forestry (km2)
Ind(21,:) = Q([460],:);

% 22. Forest area: marginal use (km2)
Ind(22,:) = Q([466],:);

% 23. Infrastructure land (km2)
Ind(23,:) = Q([465],:);

% 24. Other land use (km2)
Ind(24,:) = Q([461],:);

% 25. Energy Demand (TJ)
Ind(25,:) = sum(Q([467:470],:));

% 26. CH4 air emissions (kg)
Ind(26,:) = sum(Q([25,68:75,427,436],:));

% 27. N2O air emissions (kg)
Ind(27,:) = sum(Q([26,430],:));

% 28. CO air emissions (kg)
Ind(28,:) = sum(Q([30,76:92,437],:));

% 29. PM2.5 air emissions (kg)
Ind(29,:) = sum(Q([40,283:328,445],:));

% 30. SOx air emissions (kg)
Ind(30,:) = sum(Q([27,343:361,446],:));

% 31. NOx air emissions (kg)
Ind(31,:) = sum(Q([28,189:210,432,443],:));

% 32. NH3 air emissions (kg)
Ind(32,:) = sum(Q([29,141,431,442],:));

% 33. Benzo(a)pyrene air emissions (kg)
Ind(33,:) = sum(Q([31, 59:61],:));

% 34. Benzo(b)fluoranthene air emissions (kg)
Ind(34,:) = sum(Q([32, 62:64],:));

% 35. Benzo(k)fluoranthene air emissions (kg)
Ind(35,:) = sum(Q([33, 65:67],:));

% 36. Indeno air emissions (kg)
Ind(36,:) = sum(Q([34, 138:140],:));

% 37. PCB air emissions (kg)
Ind(37,:) = sum(Q([35, 226:230],:));

% 38. PCDD/F air emissions (kg)
Ind(38,:) = sum(Q([36, 231:236],:));

% 39. PAH air emissions (kg)
Ind(39,:) = sum(Q([219:225],:));

% 40. HCB air emissions (kg)
Ind(40,:) = sum(Q([37, 122:123, 133],:));

% 41. NMVOC air emissions (kg)
Ind(41,:) = sum(Q([38, 142:188],:));

% 42. PM10 air emissions (kg)
Ind(42,:) = sum(Q([39, 257:282],:));

% 43. TSP air emissions (kg)
Ind(43,:) = sum(Q([41, 366:411],:));

% 44. SF6 air emissions (kg)
Ind(44,:) = Q([424],:);

% 45. As air emissions (kg)
Ind(45,:) = sum(Q([42, 51:58],:));

% 46. Cd air emissions (kg)
Ind(46,:) = sum(Q([43, 95:107],:));

% 47. Cr air emissions (kg)
Ind(47,:) = sum(Q([44, 108:114],:));

% 48. Cu air emissions (kg)
Ind(48,:) = sum(Q([45, 115:121],:));

% 49. Hg air emissions (kg)
Ind(49,:) = sum(Q([46, 124:132, 134:137],:));

% 50. Ni air emissions (kg)
Ind(50,:) = sum(Q([47, 211:218],:));

% 51. Pb air emissions (kg)
Ind(51,:) = sum(Q([48, 329:342],:));

% 52. Se air emissions (kg)
Ind(52,:) = sum(Q([49, 362:365],:));

% 53. Zn air emissions (kg)
Ind(53,:) = sum(Q([50, 412:423],:));

% 54. N water emissions (kg)
Ind(54,:) = sum(Q([429, 441],:));

% 55. P water emissions (kg)
Ind(55,:) = sum(Q([434, 444],:));

% 56. Pxx soil emissions (kg)
Ind(56,:) = Q([435],:);

% 57. Value added: total (Mio. Euro)*
Ind(57,:) = sum(Q([1:9],:));

% 58. Compensation of total workforce (Mio. Euro)*
Ind(58,:) = sum(Q([3:5],:));

% 59. Compensation of low-skilled workforce (Mio. Euro)
Ind(59,:) = Q([3],:);

% 60. Compensation of medium-skilled workforce (Mio. Euro)
Ind(60,:) = Q([4],:);

% 61. Compensation of high-skilled workforce (Mio. Euro)
Ind(61,:) = Q([5],:);

% 62. Operating surplus (Mio. Euro)
Ind(62,:) = sum(Q([6:9],:));

% 63. Taxes (Mio. Euro)
Ind(63,:) = sum(Q([1:2],:));

% 64. Workforce: total (1000 people FTE)*
Ind(64,:) = sum(Q([10:15,22],:));

% 65. Workforce: Low-skilled male (1000 people FTE)
Ind(65,:) = Q([10],:);

% 66. Workforce: Low-skilled female (1000 people FTE)
Ind(66,:) = Q([11],:);

% 67. Workforce: Medium-skilled male (1000 people FTE)
Ind(67,:) = Q([12],:);

% 68. Workforce: Medium-skilled female (1000 people FTE)
Ind(68,:) = Q([13],:);

% 69. Workforce: High-skilled male (1000 people FTE)
Ind(69,:) = Q([14],:);

% 70. Workforce: High-skilled female (1000 people FTE)
Ind(70,:) = Q([15],:);

% 71. Workforce: Vulnerable Workforce (1000 people FTE)
Ind(71,:) = Q([22],:);




%% Derive the impact coefficients of the selected indicators
k_ind = 0;
for i = index_indicators;
k_ind = k_ind + 1;    
Ind(i,:) = Ind(i,:) .* (Ind(i,:)>0); 
d(k_ind,:) = Ind(i,:) ./ TotalOut';  
end

d(isinf(d))=0 ; 
d(isnan(d))=0 ;  
d = d .* (d>0); 




%% Derive all environmental and socio-economic indicators for the households (Ind_hh)
% *Indicators covered in the Global Resource Outlook (2019) and Cabernard et al. (2019)
if sum(ismember(index_output,2)) > 0; % only if global shares are calculated

% 1. Material footprint (kt)*
Ind_hh(1,:) = sum(Q_hh([471:687],:));

% 2. Unused material footprint (kt)
Ind_hh(2,:) = sum(Q_hh([688:910],:));

% 3. Climate change impacts: total (kg CO2-eQ_hh)*
CF_CH4 = 28;
CF_N2O = 265;
Ind_hh(3,:) = sum(Q_hh([24,93,94,428,438,439],:)) + sum(Q_hh([25,68:75,427,436],:)) .* CF_CH4 + sum(Q_hh([26,430],:)) .* CF_N2O + Q_hh(425,:) + Q_hh(426,:);

% 4. CO2-related climate change impacts (kg CO2-eQ_hh)
Ind_hh(4,:) = sum(Q_hh([24,93,94,428,439],:));

% 5. CH4-related climate change impacts (kg CO2-eQ_hh)
Ind_hh(5,:) = sum(Q_hh([25,68:75,427,436],:)) .* CF_CH4;

% 6. N2O-related climate change impacts (kg CO2-eQ_hh)
Ind_hh(6,:) = sum(Q_hh([26,430],:)) .* CF_N2O;

% 7. HFC-related climate change impacts (kg CO2-eQ_hh)
Ind_hh(7,:) = Q_hh([425],:);

% 8. PFC-related climate change impacts (kg CO2-eQ_hh)
Ind_hh(8,:) = Q_hh([426],:);

% 9. Particulate-matter rel. health impacts: total (DALYs)*
Ind_hh(9,:) = sum(Q_hh([40,283:328,445],:)) .* CF_PM25_other + sum(Q_hh([27,343:361,446],:)) .* CF_SOx_other + sum(Q_hh([28,189:210,432,443],:)) .* CF_NOx_other + sum(Q_hh([29,141,431,442],:)) .* CF_NH3_other;

% 10. PM2.5-related health impacts (DALYs)
Ind_hh(10,:) = sum(Q_hh([40,283:328,445],:)) .* CF_PM25_other;

% 11. SOx-related health impacts (DALYs)
Ind_hh(11,:) = sum(Q_hh([27,343:361,446],:)) .* CF_SOx_other;

% 12. NOx-related health impacts (DALYs)
Ind_hh(12,:) = sum(Q_hh([28,189:210,432,443],:)) .* CF_NOx_other;

% 13. NH3-related health impacts (DALYs)
Ind_hh(13,:) = sum(Q_hh([29,141,431,442],:)) .* CF_NH3_other;

% 14. Water Stress (Mio. m3 H2O-eQ_hh)*
% Calculate Water Stress (WS) in Mio m3 H2O eQ_hhuivalents
%Define CFs for households
n = 0;
for i = 1:49
for j =  repmat(10,1,7)
n = n+1;    
CF_water_hh(n) = CF_water(i,j);
end
end
WS_hh = sum(Q_hh([924:1026],:)); 
Ind_hh(14,:) = CF_water_hh .* WS_hh;
clear WS_hh

% 15. Blue water consumption (Mio. m3 H2O)
Ind_hh(15,:) = sum(Q_hh([924:1026],:));

% 16. Green water consumption (Mio. m3 H2O)
Ind_hh(16,:) = sum(Q_hh([911:923],:));


% 17. Land-use related biodiversity loss (global PDF)*
% Calculate Land-use related Biodiversity Loss (LU) in global PDF
% Read in data to derive CFs for land-use related-biodiversity loss
datapath = ['Input_Data/CF/'];
Q_2007_hh = dlmread([datapath 'F_hh_2007.txt'],'\t',2,1);
Q_2007_hh = Q_2007_hh(:,FinalDemand_No_Tool);

% Derive CFs for households
Ext_2007_hh(1,:) = Q_2007_hh(447,:);
Ext_2007_hh(2,:) = Q_2007_hh(448,:);
Ext_2007_hh(3,:) = sum(Q_2007_hh(449:453,:));
Ext_2007_hh(4,:) = Q_2007_hh(454,:);
Ext_2007_hh(5,:) = Q_2007_hh(455,:);
Ext_2007_hh(6,:) = Q_2007_hh(456,:);
Ext_2007_hh(7,:) = Q_2007_hh(457,:);
Ext_2007_hh(8,:) = Q_2007_hh(458,:);
Ext_2007_hh(9,:) = Q_2007_hh(459,:);
Ext_2007_hh(10,:) = Q_2007_hh(460,:);
Ext_2007_hh(11,:) = Q_2007_hh(461,:);
Ext_2007_hh(12,:) = sum(Q_2007_hh(462:464,:));
Ext_2007_hh(13,:) = Q_2007_hh(465,:);
Ext_2007_hh(14,:) = Q_2007_hh(466,:);

Ext_year_hh(1,:) = Q_hh(447,:);
Ext_year_hh(2,:) = Q_hh(448,:);
Ext_year_hh(3,:) = sum(Q_hh(449:453,:));
Ext_year_hh(4,:) = Q_hh(454,:);
Ext_year_hh(5,:) = Q_hh(455,:);
Ext_year_hh(6,:) = Q_hh(456,:);
Ext_year_hh(7,:) = Q_hh(457,:);
Ext_year_hh(8,:) = Q_hh(458,:);
Ext_year_hh(9,:) = Q_hh(459,:);
Ext_year_hh(10,:) = Q_hh(460,:);
Ext_year_hh(11,:) = Q_hh(461,:);
Ext_year_hh(12,:) = sum(Q_hh(462:464,:));
Ext_year_hh(13,:) = Q_hh(465,:);
Ext_year_hh(14,:) = Q_hh(466,:);

for q = 1:14;
for n = 0:48;
i = 1 + 7*n;
j = i + 6;
z = n+1;
CF_landuse_hh(q,i:j) = CF_landuse(z,q);
end
end

for q = 1:9
for n = 0:48;
    i = 1 + 7*n;
    j = i + 6;
for z =  i:j;  
    if Ext_2007_hh(q,z)>0
        CF_landuse_final_hh(q,z)= CF_landuse_hh(q,z) / sum(sum(Ext_2007_hh(q,i:j)));
    else
        CF_landuse_final_hh(q,z)= 0;  
    end
end
end
end

CF_landuse_final_hh(10:14,:) = CF_landuse_hh(10:14,:)*10^6;
Ind_hh(17,:) =  sum(CF_landuse_final_hh .* Ext_year_hh);


% 18. Land-use area - total (km2)
Ind_hh(18,:) = sum(Q_hh([447:466],:));

% 19. Cropland (km2)
Ind_hh(19,:) = sum(Q_hh([447:459],:));

% 20. Pemanent pastures (km2)
Ind_hh(20,:) = sum(Q_hh([462:464],:));

% 21. Forest area: forestry (km2)
Ind_hh(21,:) = Q_hh([460],:);

% 22. Forest area: marginal use (km2)
Ind_hh(22,:) = Q_hh([466],:);

% 23. Infrastructure land (km2)
Ind_hh(23,:) = Q_hh([465],:);

% 24. Other land use (km2)
Ind_hh(24,:) = Q_hh([461],:);

% 25. Energy Demand (TJ)
Ind_hh(25,:) = sum(Q_hh([467:470],:));

% 26. CH4 air emissions (kg)
Ind_hh(26,:) = sum(Q_hh([25,68:75,427,436],:));

% 27. N2O air emissions (kg)
Ind_hh(27,:) = sum(Q_hh([26,430],:));

% 28. CO air emissions (kg)
Ind_hh(28,:) = sum(Q_hh([30,76:92,437],:));

% 29. PM2.5 air emissions (kg)
Ind_hh(29,:) = sum(Q_hh([40,283:328,445],:));

% 30. SOx air emissions (kg)
Ind_hh(30,:) = sum(Q_hh([27,343:361,446],:));

% 31. NOx air emissions (kg)
Ind_hh(31,:) = sum(Q_hh([28,189:210,432,443],:));

% 32. NH3 air emissions (kg)
Ind_hh(32,:) = sum(Q_hh([29,141,431,442],:));

% 33. Benzo(a)pyrene air emissions (kg)
Ind_hh(33,:) = sum(Q_hh([31, 59:61],:));

% 34. Benzo(b)fluoranthene air emissions (kg)
Ind_hh(34,:) = sum(Q_hh([32, 62:64],:));

% 35. Benzo(k)fluoranthene air emissions (kg)
Ind_hh(35,:) = sum(Q_hh([33, 65:67],:));

% 36. Indeno air emissions (kg)
Ind_hh(36,:) = sum(Q_hh([34, 138:140],:));

% 37. PCB air emissions (kg)
Ind_hh(37,:) = sum(Q_hh([35, 226:230],:));

% 38. PCDD/F air emissions (kg)
Ind_hh(38,:) = sum(Q_hh([36, 231:236],:));

% 39. PAH air emissions (kg)
Ind_hh(39,:) = sum(Q_hh([219:225],:));

% 40. HCB air emissions (kg)
Ind_hh(40,:) = sum(Q_hh([37, 122:123, 133],:));

% 41. NMVOC air emissions (kg)
Ind_hh(41,:) = sum(Q_hh([38, 142:188],:));

% 42. PM10 air emissions (kg)
Ind_hh(42,:) = sum(Q_hh([39, 257:282],:));

% 43. TSP air emissions (kg)
Ind_hh(43,:) = sum(Q_hh([41, 366:411],:));

% 44. SF6 air emissions (kg)
Ind_hh(44,:) = Q_hh([424],:);

% 45. As air emissions (kg)
Ind_hh(45,:) = sum(Q_hh([42, 51:58],:));

% 46. Cd air emissions (kg)
Ind_hh(46,:) = sum(Q_hh([43, 95:107],:));

% 47. Cr air emissions (kg)
Ind_hh(47,:) = sum(Q_hh([44, 108:114],:));

% 48. Cu air emissions (kg)
Ind_hh(48,:) = sum(Q_hh([45, 115:121],:));

% 49. Hg air emissions (kg)
Ind_hh(49,:) = sum(Q_hh([46, 124:132, 134:137],:));

% 50. Ni air emissions (kg)
Ind_hh(50,:) = sum(Q_hh([47, 211:218],:));

% 51. Pb air emissions (kg)
Ind_hh(51,:) = sum(Q_hh([48, 329:342],:));

% 52. Se air emissions (kg)
Ind_hh(52,:) = sum(Q_hh([49, 362:365],:));

% 53. Zn air emissions (kg)
Ind_hh(53,:) = sum(Q_hh([50, 412:423],:));

% 54. N water emissions (kg)
Ind_hh(54,:) = sum(Q_hh([429, 441],:));

% 55. P water emissions (kg)
Ind_hh(55,:) = sum(Q_hh([434, 444],:));

% 56. Pxx soil emissions (kg)
Ind_hh(56,:) = Q_hh([435],:);

% 57. Value added: total (Mio. Euro)*
Ind_hh(57,:) = sum(Q_hh([1:9],:));

% 58. Compensation of total workforce (Mio. Euro)*
Ind_hh(58,:) = sum(Q_hh([3:5],:));

% 59. Compensation of low-skilled workforce (Mio. Euro)
Ind_hh(59,:) = Q_hh([3],:);

% 60. Compensation of medium-skilled workforce (Mio. Euro)
Ind_hh(60,:) = Q_hh([4],:);

% 61. Compensation of high-skilled workforce (Mio. Euro)
Ind_hh(61,:) = Q_hh([5],:);

% 62. Operating surplus (Mio. Euro)
Ind_hh(62,:) = sum(Q_hh([6:9],:));

% 63. Taxes (Mio. Euro)
Ind_hh(63,:) = sum(Q_hh([1:2],:));

% 64. Workforce: total (1000 people FTE)*
Ind_hh(64,:) = sum(Q_hh([10:15,22],:));

% 65. Workforce: Low-skilled male (1000 people FTE)
Ind_hh(65,:) = Q_hh([10],:);

% 66. Workforce: Low-skilled female (1000 people FTE)
Ind_hh(66,:) = Q_hh([11],:);

% 67. Workforce: Medium-skilled male (1000 people FTE)
Ind_hh(67,:) = Q_hh([12],:);

% 68. Workforce: Medium-skilled female (1000 people FTE)
Ind_hh(68,:) = Q_hh([13],:);

% 69. Workforce: High-skilled male (1000 people FTE)
Ind_hh(69,:) = Q_hh([14],:);

% 70. Workforce: High-skilled female (1000 people FTE)
Ind_hh(70,:) = Q_hh([15],:);

% 71. Workforce: Vulnerable Workforce (1000 people FTE)
Ind_hh(71,:) = Q_hh([22],:);
end






%% Calculate total global impacts
if sum(ismember(index_output,2)) > 0; % only if global shares are calculated
k_ind = 0;
for i = index_indicators;
k_ind = k_ind + 1;    
Total_Global_Impacts(k_ind,k_time) = sum(Ind(i,:)) + sum(Ind_hh(i,:));
end
end






%% Calculate the scope 3 impacts of target-sector-regions without double counting for each indicator
k_ind = 0;     
for i = index_indicators   
k_ind = k_ind + 1;      

% Total scope 3 impacts of target-sector-regions
TOT_ind_year(k_ind,k_time) = d(k_ind,:) * L(:,index_t) * x_t_wdc;

% Calculate the scope 3 impacts of target-sector-regions from different
% perspectives and map the linkages within these perspectives
if sum(ismember(index_results,[1:39])) > 0;

% 2D-array: links between production (rows) and target perspective (columns)
if sum(ismember(index_results,[1,2,5,9,11,16,17,20,21,23,24,28,29,32,33,35,36])) > 0;
P_vs_T_all = diag(d(k_ind,:)) * L(:,index_t) * diag(sum(X_t_wdc_C,2)); 
end

% 2D-array: links between target (rows) and final supply perspective (columns)
if sum(ismember(index_results,[3,6,10,18,22,25,30,34,37])) > 0;
T_vs_FS_all = diag(d(k_ind,:) * L(:,index_t)) * X_t_wdc_O; 
end

% 2D-array: links between target (rows) and final demand perspective (columns) 
if sum(ismember(index_results,[4,7,12,14,19,26,27,31,38,39])) > 0;
T_vs_FD_all = diag(d(k_ind,:) * L(:,index_t)) * X_t_wdc_C; 
end

% 2D-array: links of E_T_wdc between production (rows) and final demand perspective (column)
if sum(ismember(index_results,[8,13,15])) > 0;
P_vs_FD_all = diag(d(k_ind,:)) * L(:,index_t) * X_t_wdc_C; 
end 



%% Compile Perspectives

% Compile Production Perspective for each indicator and year
if sum(ismember(index_results,[1,16,20,23,28,32,35])) > 0;
P_all_ind_year(:,k_ind,k_time) = sum(P_vs_T_all,2);
P_all_mat = vec2mat(P_all_ind_year(:,k_ind,k_time),163)';
P_sectors_ind_year(:,k_ind,k_time) = sum(P_all_mat,2);
P_regions_ind_year(:,k_ind,k_time) = sum(P_all_mat)';
end

% Compile Target Perspective for each indicator and year
if sum(ismember(index_results,[2,17,21,24,29,33,36])) > 0;
T_all_ind_year(:,k_ind,k_time) = sum(P_vs_T_all,1)';
T_all_mat = vec2mat(T_all_ind_year(:,k_ind,k_time),n_t_s)';
T_sectors_ind_year(:,k_ind,k_time) = sum(T_all_mat,2);
T_regions_ind_year(:,k_ind,k_time) = sum(T_all_mat,1)';
end

% Compile Final Supply Perspective for each indicator and year
if sum(ismember(index_results,[3,18,22,25,30,34,37])) > 0;
FS_all_T(:,1) = T_vs_FS_all(:,1);
FS_all_O(:,1) = sum(T_vs_FS_all(:,2:(n_o+1)),1)';
k = 0;
for z = index_t
    k = k+1;
FS_all_ind_year(z,k_ind,k_time) = FS_all_T(k);
end
k = 0;
for z = index_o
    k = k+1;
FS_all_ind_year(z,k_ind,k_time) = FS_all_O(k);
end

FS_all_mat = vec2mat(FS_all_ind_year(:,k_ind,k_time),163)';
FS_sectors_ind_year(:,k_ind,k_time) = sum(FS_all_mat,2);
FS_regions_ind_year(:,k_ind,k_time) = sum(FS_all_mat,1)';
end

% Compile Final Demand Perspective for each indicator and year
if sum(ismember(index_results,[4,19,26,27,31,38,39])) > 0;
FD_all_ind_year(:,k_ind,k_time) = sum(T_vs_FD_all,1)';
FD_all_mat = vec2mat(FD_all_ind_year(:,k_ind,k_time),7)';
FD_cat_ind_year(:,k_ind,k_time) = sum(FD_all_mat,2);
FD_regions_ind_year(:,k_ind,k_time) = sum(FD_all_mat)';
end





%% Compile and Save Linkages between Perspectives as text files (one table for each year and indicator):
if sum(ismember(index_results,[5:15])) > 0; 
indicator_name = char(Labels_Indicators(i));

%% Compile and save linkages in the unit of the indicator
if sum(ismember(index_output,1)) > 0;    % output in the unit of the indicator
    
mkdir(['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '']); 

TOT = TOT_ind_year(k_ind,k_time);

% All linkages between Production and Target 
if sum(ismember(index_results,5)) > 0;
P_vs_T_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - All Linkages between Production (Rows) and Target Perspective (Columns) - Upstream supply chain'});
P_vs_T_all_table(2:7988,1) = Labels_Production_Tool;
P_vs_T_all_table(1,2:(n_t+1)) = Labels_Target_Tool;
P_vs_T_all_table(2:7988,2:(n_t+1)) = num2cell(P_vs_T_all); 

P_vs_T_all_table(7989,1) = {'Total per Target-Sector-Region'};
P_vs_T_all_table(7989,2:(n_t+1)) = num2cell(sum(P_vs_T_all,1));

P_vs_T_all_table(1,(n_t+2)) = {'Total per Sector and Region of Production'};
P_vs_T_all_table(2:7988,(n_t+2)) = num2cell(sum(P_vs_T_all,2));
P_vs_T_all_table(7989,(n_t+2)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_T_all_table),[datapath 'All_Linkages_Production_vs_Target_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% All linkages between Target and Final Supply
if sum(ismember(index_results,6)) > 0;
T_vs_FS_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - All Linkages between Target (Rows) and Final Supply Perspective (Columns) - Downstream value chain'});
T_vs_FS_all_table(2:(n_t+1),1) = Labels_Target_Tool;
T_vs_FS_all_table(1,2:(n_o+2)) = Labels_FinalSupply_Tool;
T_vs_FS_all_table(2:(n_t+1),2:(n_o+2)) = num2cell(T_vs_FS_all); 

T_vs_FS_all_table((n_t+2),1) = {'Total per Non-Target-Sector-Region'};
T_vs_FS_all_table((n_t+2),2:(n_o+2)) = num2cell(sum(T_vs_FS_all,1));

T_vs_FS_all_table(1,(n_o+3)) = {'Total per Target-Sector-Region'};
T_vs_FS_all_table(2:(n_t+1),(n_o+3)) = num2cell(sum(T_vs_FS_all,2));
T_vs_FS_all_table((n_t+2),(n_o+3)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FS_all_table),[datapath 'All_Linkages_Target_vs_FinalSupply_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% All linkages between Target and Final Demand
if sum(ismember(index_results,7)) > 0;
T_vs_FD_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - All Linkages between Target (Rows) and Final Demand Perspective (Columns) - Dowstream value chain'});
T_vs_FD_all_table(2:(n_t+1),1) = Labels_Target_Tool;
T_vs_FD_all_table(1,2:344) = Labels_FinalDemand_Tool;
T_vs_FD_all_table(2:(n_t+1),2:344) = num2cell(T_vs_FD_all); 

T_vs_FD_all_table((n_t+2),1) = {'Total per Region and Category of Final Demand'};
T_vs_FD_all_table((n_t+2),2:344) = num2cell(sum(T_vs_FD_all,1));

T_vs_FD_all_table(1,345) = {'Total per Target-Sector-Region'};
T_vs_FD_all_table(2:(n_t+1),345) = num2cell(sum(T_vs_FD_all,2));
T_vs_FD_all_table((n_t+2),345) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FD_all_table),[datapath 'All_Linkages_Target_vs_FinalDemand_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% All linkages between Production and Final Demand
if sum(ismember(index_results,8)) > 0;
P_vs_FD_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - All Linkages between Production (Rows) and Final Demand Perspective (Columns) - Entire value chain'});
P_vs_FD_all_table(2:7988,1) = Labels_Production_Tool;
P_vs_FD_all_table(1,2:344) = Labels_FinalDemand_Tool;
P_vs_FD_all_table(2:7988,2:344) = num2cell(P_vs_FD_all); 

P_vs_FD_all_table(7989,1) = {'Total per Region and Category of Final Demand'};
P_vs_FD_all_table(7989,2:344) = num2cell(sum(P_vs_FD_all));

P_vs_FD_all_table(1,345) = {'Total per Sector and Region of Production'};
P_vs_FD_all_table(2:7988,345) = num2cell(sum(P_vs_FD_all,2));
P_vs_FD_all_table(7989,345) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_FD_all_table),[datapath 'All_Linkages_Production_vs_FinalDemand_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Sectoral linkages between Production and Target 
if sum(ismember(index_results,9)) > 0;
for n = 1:163
P_vs_T_S1(n,:) = sum(P_vs_T_all(index_matrix_P(:,n),:),1);
end

for n = 1:n_t_s
P_vs_T_S2(:,n) = sum(P_vs_T_S1(:,index_matrix_T(:,n)),2);
end

P_vs_T_sectors_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Sectoral Linkages between Production (Rows) and Target Perspective (Columns) - Upstream supply chain'});
P_vs_T_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
P_vs_T_sectors_table(1,2:(n_t_s+1)) = Labels_Target_Sectors_Tool;
P_vs_T_sectors_table(2:164,2:(n_t_s+1)) = num2cell(P_vs_T_S2);

P_vs_T_sectors_table(165,1) = {'Total per Target-Sector'};
P_vs_T_sectors_table(165,2:(n_t_s+1)) = num2cell(sum(P_vs_T_S2,1));

P_vs_T_sectors_table(1,(n_t_s+2)) = {'Total per Sector of Production'};
P_vs_T_sectors_table(2:164,(n_t_s+2)) = num2cell(sum(P_vs_T_S2,2));
P_vs_T_sectors_table(165,n_t_s+2) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_T_sectors_table),[datapath 'Sectoral_Linkages_Production_vs_Target_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Sectoral linkages between Target and Final Supply 
if sum(ismember(index_results,10)) > 0;
for n = 1:n_t_s
T_vs_FS_S1(n,:) = sum(T_vs_FS_all(index_matrix_T(:,n),:),1);
end

T_vs_FS_S1_T = T_vs_FS_S1(:,1);
T_vs_FS_S1_O = T_vs_FS_S1(:,2:(n_o+1));

T_vs_FS_S2_O = zeros(n_t_s,7987);

k = 0;
for n = index_o
k = k + 1;
T_vs_FS_S2_O(:,n) = T_vs_FS_S1_O(:,k);
end

for n = 1:163
T_vs_FS_S3_O(:,n) = sum(T_vs_FS_S2_O(:,index_matrix_P(:,n)),2);
end

T_vs_FS_S2(:,1) = T_vs_FS_S1_T;
T_vs_FS_S2(:,2:(n_o_s+1)) = T_vs_FS_S3_O(:,index_o_s);

T_vs_FS_sectors_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Sectoral Linkages between Target (Rows) and Final Supply Perspective (Columns) - Downstream value chain'});
T_vs_FS_sectors_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
T_vs_FS_sectors_table(1,2) = Labels_FinalSupply_Tool(1);
T_vs_FS_sectors_table(1,3:(n_o_s+2)) = Labels_Sectors_all_Tool(index_o_s);
T_vs_FS_sectors_table(2:(n_t_s+1),2:(n_o_s+2)) = num2cell(T_vs_FS_S2); 

T_vs_FS_sectors_table((n_t_s+2),1) = {'Total per Non-Target-Sector'};
T_vs_FS_sectors_table((n_t_s+2),2:(n_o_s+2)) = num2cell(sum(T_vs_FS_S2,1));

T_vs_FS_sectors_table(1,(n_o_s+3)) = {'Total per Target-Sector'};
T_vs_FS_sectors_table(2:(n_t_s+1),(n_o_s+3)) = num2cell(sum(T_vs_FS_S2,2));
T_vs_FS_sectors_table((n_t_s+2),(n_o_s+3)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FS_sectors_table),[datapath 'Sectoral_Linkages_Target_vs_FinalSupply_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Regional linkages between Production and Target 
if sum(ismember(index_results,11)) > 0;
for n = 1:49
P_vs_T_R1(n,:) = sum(P_vs_T_all(index_matrix_P(n,:),:),1);
end

for n = 1:n_t_r
P_vs_T_R2(:,n) = sum(P_vs_T_R1(:,index_matrix_T(n,:)),2);
end

P_vs_T_regions_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Regional Linkages between Production (Rows) and Target Perspective (Columns) - Upstream supply chain'});
P_vs_T_regions_table(2:50,1) = Labels_Regions_all_Tool;
P_vs_T_regions_table(1,2:(n_t_r+1)) = Labels_Target_Regions_Tool;
P_vs_T_regions_table(2:50,2:(n_t_r+1)) = num2cell(P_vs_T_R2);

P_vs_T_regions_table(51,1) = {'Total per Target-Region'};
P_vs_T_regions_table(51,2:(n_t_r+1)) = num2cell(sum(P_vs_T_R2,1));

P_vs_T_regions_table(1,(n_t_r+2)) = {'Total per Region of Production'};
P_vs_T_regions_table(2:50,(n_t_r+2)) = num2cell(sum(P_vs_T_R2,2));
P_vs_T_regions_table(51,n_t_r+2) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_T_regions_table),[datapath 'Regional_Linkages_Production_vs_Target_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  


% Regional linkages between Target and Final Demand 
if sum(ismember(index_results,12)) > 0;
for n = 1:n_t_r
T_vs_FD_R1(n,:) = sum(T_vs_FD_all(index_matrix_T(n,:),:),1);
end

for n = 1:49
T_vs_FD_R2(:,n) = sum(T_vs_FD_R1(:,index_matrix_FD(n,:)),2);
end

T_vs_FD_regions_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Regional Linkages between Target (Rows) and Final Demand Perspective (Columns) - Downstream value chain)'});
T_vs_FD_regions_table(2:(n_t_r+1),1) = Labels_Target_Regions_Tool;
T_vs_FD_regions_table(1,2:50) = Labels_Regions_all_Tool;
T_vs_FD_regions_table(2:(n_t_r+1),2:50) = num2cell(T_vs_FD_R2);

T_vs_FD_regions_table((n_t_r+2),1) = {'Total per Region of Final Demand'};
T_vs_FD_regions_table((n_t_r+2),2:50) = num2cell(sum(T_vs_FD_R2,1));

T_vs_FD_regions_table(1,51) = {'Total per Target-Region'};
T_vs_FD_regions_table(2:(n_t_r+1),51) = num2cell(sum(T_vs_FD_R2,2));
T_vs_FD_regions_table((n_t_r+2),51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FD_regions_table),[datapath 'Regional_Linkages_Target_vs_FinalDemand_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  


% Regional linkages between Production and Final Demand 
if sum(ismember(index_results,13)) > 0;
for n = 1:49
P_vs_FD_R1(n,:) = sum(P_vs_FD_all(index_matrix_P(n,:),:));
end

for n = 1:49
P_vs_FD_R2(:,n) = sum(P_vs_FD_R1(:,index_matrix_FD(n,:)),2);
end

P_vs_FD_regions_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Regional Linkages between Production (Rows) and Final Demand Perspective (Columns) - Entire value chain'});
P_vs_FD_regions_table(2:50,1) = Labels_Regions_all_Tool;
P_vs_FD_regions_table(1,2:50) = Labels_Regions_all_Tool;
P_vs_FD_regions_table(2:50,2:50) = num2cell(P_vs_FD_R2);

P_vs_FD_regions_table(51,1) = {'Total per Region of Final Demand'};
P_vs_FD_regions_table(51,2:50) = num2cell(sum(P_vs_FD_R2));

P_vs_FD_regions_table(1,51) = {'Total per Region of Production'};
P_vs_FD_regions_table(2:50,51) = num2cell(sum(P_vs_FD_R2,2));
P_vs_FD_regions_table(51,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_FD_regions_table),[datapath 'Regional_Linkages_Production_vs_FinalDemand_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end   


% Linkages between Target-Sectors and Final-Demand-Regions
if sum(ismember(index_results,14)) > 0;
for n = 1:n_t_s
T_sec_vs_FD(n,:) = sum(T_vs_FD_all(index_matrix_T(:,n),:),1);
end

for n = 1:49
T_sec_vs_FD_reg(:,n) = sum(T_sec_vs_FD(:,index_matrix_FD(n,:)),2);
end

T_sec_vs_FD_reg_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Linkages between Target-Sector (Rows) and Region of Final Demand (Columns) - Downstream value chain'});
T_sec_vs_FD_reg_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
T_sec_vs_FD_reg_table(1,2:50) = Labels_Regions_all_Tool;
T_sec_vs_FD_reg_table(2:(n_t_s+1),2:50) = num2cell(T_sec_vs_FD_reg);

T_sec_vs_FD_reg_table((n_t_s+2),1) = {'Total per Region of Final Demand'};
T_sec_vs_FD_reg_table((n_t_s+2),2:50) = num2cell(sum(T_sec_vs_FD_reg,1));

T_sec_vs_FD_reg_table(1,51) = {'Total per Target-Sector'};
T_sec_vs_FD_reg_table(2:(n_t_s+1),51) = num2cell(sum(T_sec_vs_FD_reg,2));
T_sec_vs_FD_reg_table((n_t_s+2),51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_sec_vs_FD_reg_table),[datapath 'Linkages_Target_Sectors_vs_FinalDemand_Regions_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end    


% Linkages between Production-Sectors and Final-Demand-Regions
if sum(ismember(index_results,15)) > 0;

for n = 1:163
P_sec_vs_FD(n,:) = sum(P_vs_FD_all(index_matrix_P(:,n),:));
end

for n = 1:49
P_sec_vs_FD_reg(:,n) = sum(P_sec_vs_FD(:,index_matrix_FD(n,:)),2);
end

P_sec_vs_FD_reg_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Linkages between Producing Sector (Rows) and Region of Final Demand (Columns) - Entire value chain'});
P_sec_vs_FD_reg_table(2:164,1) = Labels_Sectors_all_Tool;
P_sec_vs_FD_reg_table(1,2:50) = Labels_Regions_all_Tool;
P_sec_vs_FD_reg_table(2:164,2:50) = num2cell(P_sec_vs_FD_reg);

P_sec_vs_FD_reg_table(165,1) = {'Total per Region of Final Demand'};
P_sec_vs_FD_reg_table(165,2:50) = num2cell(sum(P_sec_vs_FD_reg));

P_sec_vs_FD_reg_table(1,51) = {'Total per Sector'};
P_sec_vs_FD_reg_table(2:164,51) = num2cell(sum(P_sec_vs_FD_reg,2));
P_sec_vs_FD_reg_table(165,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_sec_vs_FD_reg_table),[datapath 'Linkages_Production_Sectors_vs_FinalDemand_Regions_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

end % End output in the unit of the indicator




%% Compile and save linkages as shares in total global impact
if sum(ismember(index_output,2)) > 0; % Output in global shares

mkdir(['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '']); 

TOT = TOT_ind_year(k_ind,k_time) / Total_Global_Impacts(k_ind,k_time);

% All linkages between Production and Target 
if sum(ismember(index_results,5)) > 0;
P_vs_T_all_share = P_vs_T_all ./ Total_Global_Impacts(k_ind,k_time);    
P_vs_T_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - All Linkages between Production (Rows) and Target Perspective (Columns) - Upstream supply chain'});
P_vs_T_all_table(2:7988,1) = Labels_Production_Tool;
P_vs_T_all_table(1,2:(n_t+1)) = Labels_Target_Tool;
P_vs_T_all_table(2:7988,2:(n_t+1)) = num2cell(P_vs_T_all_share); 

P_vs_T_all_table(7989,1) = {'Total per Target-Sector-Region'};
P_vs_T_all_table(7989,2:(n_t+1)) = num2cell(sum(P_vs_T_all_share,1));

P_vs_T_all_table(1,(n_t+2)) = {'Total per Sector and Region of Production'};
P_vs_T_all_table(2:7988,(n_t+2)) = num2cell(sum(P_vs_T_all_share,2));
P_vs_T_all_table(7989,(n_t+2)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_T_all_table),[datapath 'All_Linkages_Production_vs_Target_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% All linkages between Target and Final Supply
if sum(ismember(index_results,6)) > 0;
T_vs_FS_all_share = T_vs_FS_all ./ Total_Global_Impacts(k_ind,k_time);    
T_vs_FS_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - All Linkages between Target (Rows) and Final Supply Perspective (Columns) - Downstream value chain'});
T_vs_FS_all_table(2:(n_t+1),1) = Labels_Target_Tool;
T_vs_FS_all_table(1,2:(n_o+2)) = Labels_FinalSupply_Tool;
T_vs_FS_all_table(2:(n_t+1),2:(n_o+2)) = num2cell(T_vs_FS_all_share); 

T_vs_FS_all_table((n_t+2),1) = {'Total per Non-Target-Sector-Region'};
T_vs_FS_all_table((n_t+2),2:(n_o+2)) = num2cell(sum(T_vs_FS_all_share,1));

T_vs_FS_all_table(1,(n_o+3)) = {'Total per Target-Sector-Region'};
T_vs_FS_all_table(2:(n_t+1),(n_o+3)) = num2cell(sum(T_vs_FS_all_share,2));
T_vs_FS_all_table((n_t+2),(n_o+3)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FS_all_table),[datapath 'All_Linkages_Target_vs_FinalSupply_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% All linkages between Target and Final Demand
if sum(ismember(index_results,7)) > 0;
T_vs_FD_all_share = T_vs_FD_all ./ Total_Global_Impacts(k_ind,k_time);     
T_vs_FD_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - All Linkages between Target (Rows) and Final Demand Perspective (Columns) - Dowstream value chain'});
T_vs_FD_all_table(2:(n_t+1),1) = Labels_Target_Tool;
T_vs_FD_all_table(1,2:344) = Labels_FinalDemand_Tool;
T_vs_FD_all_table(2:(n_t+1),2:344) = num2cell(T_vs_FD_all_share); 

T_vs_FD_all_table((n_t+2),1) = {'Total per Region and Category of Final Demand'};
T_vs_FD_all_table((n_t+2),2:344) = num2cell(sum(T_vs_FD_all_share,1));

T_vs_FD_all_table(1,345) = {'Total per Target-Sector-Region'};
T_vs_FD_all_table(2:(n_t+1),345) = num2cell(sum(T_vs_FD_all_share,2));
T_vs_FD_all_table((n_t+2),345) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FD_all_table),[datapath 'All_Linkages_Target_vs_FinalDemand_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% All linkages between Production and Final Demand
if sum(ismember(index_results,8)) > 0;
P_vs_FD_all_share = P_vs_FD_all ./ Total_Global_Impacts(k_ind,k_time);      
P_vs_FD_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - All Linkages between Production (Rows) and Final Demand Perspective (Columns) - Entire value chain'});
P_vs_FD_all_table(2:7988,1) = Labels_Production_Tool;
P_vs_FD_all_table(1,2:344) = Labels_FinalDemand_Tool;
P_vs_FD_all_table(2:7988,2:344) = num2cell(P_vs_FD_all_share); 

P_vs_FD_all_table(7989,1) = {'Total per Region and Category of Final Demand'};
P_vs_FD_all_table(7989,2:344) = num2cell(sum(P_vs_FD_all_share));

P_vs_FD_all_table(1,345) = {'Total per Sector and Region of Production'};
P_vs_FD_all_table(2:7988,345) = num2cell(sum(P_vs_FD_all_share,2));
P_vs_FD_all_table(7989,345) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_FD_all_table),[datapath 'All_Linkages_Production_vs_FinalDemand_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Sectoral linkages between Production and Target 
if sum(ismember(index_results,9)) > 0;
for n = 1:163
P_vs_T_S1(n,:) = sum(P_vs_T_all(index_matrix_P(:,n),:),1);
end

for n = 1:n_t_s
P_vs_T_S2(:,n) = sum(P_vs_T_S1(:,index_matrix_T(:,n)),2);
end

P_vs_T_S2 = P_vs_T_S2 ./ Total_Global_Impacts(k_ind,k_time); 
P_vs_T_sectors_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Sectoral Linkages between Production (Rows) and Target Perspective (Columns) - Upstream supply chain'});
P_vs_T_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
P_vs_T_sectors_table(1,2:(n_t_s+1)) = Labels_Target_Sectors_Tool;
P_vs_T_sectors_table(2:164,2:(n_t_s+1)) = num2cell(P_vs_T_S2);

P_vs_T_sectors_table(165,1) = {'Total per Target-Sector'};
P_vs_T_sectors_table(165,2:(n_t_s+1)) = num2cell(sum(P_vs_T_S2,1));

P_vs_T_sectors_table(1,(n_t_s+2)) = {'Total per Sector of Production'};
P_vs_T_sectors_table(2:164,(n_t_s+2)) = num2cell(sum(P_vs_T_S2,2));
P_vs_T_sectors_table(165,n_t_s+2) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_T_sectors_table),[datapath 'Sectoral_Linkages_Production_vs_Target_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Sectoral linkages between Target and Final Supply 
if sum(ismember(index_results,10)) > 0;
for n = 1:n_t_s
T_vs_FS_S1(n,:) = sum(T_vs_FS_all(index_matrix_T(:,n),:),1);
end

T_vs_FS_S1_T = T_vs_FS_S1(:,1);
T_vs_FS_S1_O = T_vs_FS_S1(:,2:(n_o+1));

T_vs_FS_S2_O = zeros(n_t_s,7987);

k = 0;
for n = index_o
k = k + 1;
T_vs_FS_S2_O(:,n) = T_vs_FS_S1_O(:,k);
end

for n = 1:163
T_vs_FS_S3_O(:,n) = sum(T_vs_FS_S2_O(:,index_matrix_P(:,n)),2);
end

T_vs_FS_S2(:,1) = T_vs_FS_S1_T;
T_vs_FS_S2(:,2:(n_o_s+1)) = T_vs_FS_S3_O(:,index_o_s);

T_vs_FS_S2 = T_vs_FS_S2 ./ Total_Global_Impacts(k_ind,k_time);

T_vs_FS_sectors_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Sectoral Linkages between Target (Rows) and Final Supply Perspective (Columns) - Downstream value chain'});
T_vs_FS_sectors_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
T_vs_FS_sectors_table(1,2) = Labels_FinalSupply_Tool(1);
T_vs_FS_sectors_table(1,3:(n_o_s+2)) = Labels_Sectors_all_Tool(index_o_s);
T_vs_FS_sectors_table(2:(n_t_s+1),2:(n_o_s+2)) = num2cell(T_vs_FS_S2); 

T_vs_FS_sectors_table((n_t_s+2),1) = {'Total per Non-Target-Sector'};
T_vs_FS_sectors_table((n_t_s+2),2:(n_o_s+2)) = num2cell(sum(T_vs_FS_S2,1));

T_vs_FS_sectors_table(1,(n_o_s+3)) = {'Total per Target-Sector'};
T_vs_FS_sectors_table(2:(n_t_s+1),(n_o_s+3)) = num2cell(sum(T_vs_FS_S2,2));
T_vs_FS_sectors_table((n_t_s+2),(n_o_s+3)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FS_sectors_table),[datapath 'Sectoral_Linkages_Target_vs_FinalSupply_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Regional linkages between Production and Target 
if sum(ismember(index_results,11)) > 0;
for n = 1:49
P_vs_T_R1(n,:) = sum(P_vs_T_all(index_matrix_P(n,:),:),1);
end

for n = 1:n_t_r
P_vs_T_R2(:,n) = sum(P_vs_T_R1(:,index_matrix_T(n,:)),2);
end

P_vs_T_R2 = P_vs_T_R2 ./ Total_Global_Impacts(k_ind,k_time);
P_vs_T_regions_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Regional Linkages between Production (Rows) and Target Perspective (Columns) - Upstream supply chain'});
P_vs_T_regions_table(2:50,1) = Labels_Regions_all_Tool;
P_vs_T_regions_table(1,2:(n_t_r+1)) = Labels_Target_Regions_Tool;
P_vs_T_regions_table(2:50,2:(n_t_r+1)) = num2cell(P_vs_T_R2);

P_vs_T_regions_table(51,1) = {'Total per Target-Region'};
P_vs_T_regions_table(51,2:(n_t_r+1)) = num2cell(sum(P_vs_T_R2,1));

P_vs_T_regions_table(1,(n_t_r+2)) = {'Total per Region of Production'};
P_vs_T_regions_table(2:50,(n_t_r+2)) = num2cell(sum(P_vs_T_R2,2));
P_vs_T_regions_table(51,n_t_r+2) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_T_regions_table),[datapath 'Regional_Linkages_Production_vs_Target_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  


% Regional linkages between Target and Final Demand 
if sum(ismember(index_results,12)) > 0;
for n = 1:n_t_r
T_vs_FD_R1(n,:) = sum(T_vs_FD_all(index_matrix_T(n,:),:),1);
end

for n = 1:49
T_vs_FD_R2(:,n) = sum(T_vs_FD_R1(:,index_matrix_FD(n,:)),2);
end
T_vs_FD_R2 = T_vs_FD_R2 ./ Total_Global_Impacts(k_ind,k_time);
T_vs_FD_regions_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Regional Linkages between Target (Rows) and Final Demand Perspective (Columns) - Downstream value chain)'});
T_vs_FD_regions_table(2:(n_t_r+1),1) = Labels_Target_Regions_Tool;
T_vs_FD_regions_table(1,2:50) = Labels_Regions_all_Tool;
T_vs_FD_regions_table(2:(n_t_r+1),2:50) = num2cell(T_vs_FD_R2);

T_vs_FD_regions_table((n_t_r+2),1) = {'Total per Region of Final Demand'};
T_vs_FD_regions_table((n_t_r+2),2:50) = num2cell(sum(T_vs_FD_R2,1));

T_vs_FD_regions_table(1,51) = {'Total per Target-Region'};
T_vs_FD_regions_table(2:(n_t_r+1),51) = num2cell(sum(T_vs_FD_R2,2));
T_vs_FD_regions_table((n_t_r+2),51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_vs_FD_regions_table),[datapath 'Regional_Linkages_Target_vs_FinalDemand_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  


% Regional linkages between Production and Final Demand 
if sum(ismember(index_results,13)) > 0;
for n = 1:49
P_vs_FD_R1(n,:) = sum(P_vs_FD_all(index_matrix_P(n,:),:));
end

for n = 1:49
P_vs_FD_R2(:,n) = sum(P_vs_FD_R1(:,index_matrix_FD(n,:)),2);
end

P_vs_FD_R2 = P_vs_FD_R2 ./ Total_Global_Impacts(k_ind,k_time);
P_vs_FD_regions_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Regional Linkages between Production (Rows) and Final Demand Perspective (Columns) - Entire value chain'});
P_vs_FD_regions_table(2:50,1) = Labels_Regions_all_Tool;
P_vs_FD_regions_table(1,2:50) = Labels_Regions_all_Tool;
P_vs_FD_regions_table(2:50,2:50) = num2cell(P_vs_FD_R2);

P_vs_FD_regions_table(51,1) = {'Total per Region of Final Demand'};
P_vs_FD_regions_table(51,2:50) = num2cell(sum(P_vs_FD_R2));

P_vs_FD_regions_table(1,51) = {'Total per Region of Production'};
P_vs_FD_regions_table(2:50,51) = num2cell(sum(P_vs_FD_R2,2));
P_vs_FD_regions_table(51,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_vs_FD_regions_table),[datapath 'Regional_Linkages_Production_vs_FinalDemand_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end   


% Linkages between Target-Sectors and Final-Demand-Regions
if sum(ismember(index_results,14)) > 0;
for n = 1:n_t_s
T_sec_vs_FD(n,:) = sum(T_vs_FD_all(index_matrix_T(:,n),:),1);
end

for n = 1:49
T_sec_vs_FD_reg(:,n) = sum(T_sec_vs_FD(:,index_matrix_FD(n,:)),2);
end

T_sec_vs_FD_reg = T_sec_vs_FD_reg ./ Total_Global_Impacts(k_ind,k_time);
T_sec_vs_FD_reg_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Linkages between Target-Sector (Rows) and Region of Final Demand (Columns) - Downstream value chain'});
T_sec_vs_FD_reg_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
T_sec_vs_FD_reg_table(1,2:50) = Labels_Regions_all_Tool;
T_sec_vs_FD_reg_table(2:(n_t_s+1),2:50) = num2cell(T_sec_vs_FD_reg);

T_sec_vs_FD_reg_table((n_t_s+2),1) = {'Total per Region of Final Demand'};
T_sec_vs_FD_reg_table((n_t_s+2),2:50) = num2cell(sum(T_sec_vs_FD_reg,1));

T_sec_vs_FD_reg_table(1,51) = {'Total per Target-Sector'};
T_sec_vs_FD_reg_table(2:(n_t_s+1),51) = num2cell(sum(T_sec_vs_FD_reg,2));
T_sec_vs_FD_reg_table((n_t_s+2),51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_sec_vs_FD_reg_table),[datapath 'Linkages_Target_Sectors_vs_FinalDemand_Regions_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end    


% Linkages between Production-Sectors and Final-Demand-Regions
if sum(ismember(index_results,15)) > 0;

for n = 1:163
P_sec_vs_FD(n,:) = sum(P_vs_FD_all(index_matrix_P(:,n),:));
end

for n = 1:49
P_sec_vs_FD_reg(:,n) = sum(P_sec_vs_FD(:,index_matrix_FD(n,:)),2);
end

P_sec_vs_FD_reg = P_sec_vs_FD_reg ./ Total_Global_Impacts(k_ind,k_time);
P_sec_vs_FD_reg_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Linkages between Producing Sector (Rows) and Region of Final Demand (Columns) - Entire value chain'});
P_sec_vs_FD_reg_table(2:164,1) = Labels_Sectors_all_Tool;
P_sec_vs_FD_reg_table(1,2:50) = Labels_Regions_all_Tool;
P_sec_vs_FD_reg_table(2:164,2:50) = num2cell(P_sec_vs_FD_reg);

P_sec_vs_FD_reg_table(165,1) = {'Total per Region of Final Demand'};
P_sec_vs_FD_reg_table(165,2:50) = num2cell(sum(P_sec_vs_FD_reg));

P_sec_vs_FD_reg_table(1,51) = {'Total per Sector'};
P_sec_vs_FD_reg_table(2:164,51) = num2cell(sum(P_sec_vs_FD_reg,2));
P_sec_vs_FD_reg_table(165,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Linkages/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_sec_vs_FD_reg_table),[datapath 'Linkages_Production_Sectors_vs_FinalDemand_Regions_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end
    
end % End output in global shares

end % End compilation linkages

end % End of the calculation for index_results = 1:39

end % End of the calculation for each indicator

end % End of the Calculation for each year







%% Compile and Save Single Perspectives as text files (one table for each year and indicator): 
if sum(ismember(index_results,[1:4])) > 0; 

% Procedure for each year    
k_time = 0;
for year = index_years;  
% increase the index for the timelines
k_time = k_time + 1;
    
k_ind = 0;
for i = index_indicators    
k_ind = k_ind + 1;    
indicator_name = char(Labels_Indicators(i)); 

%% Compile and save single perspectives in the unit of the indicator
if sum(ismember(index_output,1)) > 0;    % output in the unit of the indicator

mkdir(['' folder_name '/Results_in_Unit_of_Indicator/Single_Perspectives/Year_' int2str(year) '/' indicator_name '']);   

TOT = TOT_ind_year(k_ind,k_time);

% Single Production Perspective
if sum(ismember(index_results,[1])) > 0;
P_all(:,1) = P_all_ind_year(:,k_ind,k_time);
P_all_mat = vec2mat(P_all,163)';
P_sectors = sum(P_all_mat,2);
P_regions = sum(P_all_mat)';

P_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Production Perspective: Impacts are allocated to the sectors (rows) and regions (columns) where they are caused '});
P_all_table(2:164,1) = Labels_Sectors_all_Tool;
P_all_table(1,2:50) = Labels_Regions_all_Tool;
P_all_table(2:164,2:50) = num2cell(P_all_mat);

P_all_table(165,1) = {'Total per Region of Production'};
P_all_table(165,2:50) = num2cell(P_regions');

P_all_table(1,51) = {'Total per Sector of Production'};
P_all_table(2:164,51) = num2cell(P_sectors);
P_all_table(165,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_all_table),[datapath 'Production_Perspective_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% Single Target Perspective
if sum(ismember(index_results,2)) > 0;
T_all(:,1) = T_all_ind_year(:,k_ind,k_time);    
T_all_mat = vec2mat(T_all,n_t_s)';
T_sectors = sum(T_all_mat,2);
T_regions = sum(T_all_mat,1)';
    
T_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Target Perspective: Impacts are allocated to the target-sectors (rows) and target-regions (columns) that are finally supplied'});
T_all_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
T_all_table(1,2:(n_t_r+1)) = Labels_Target_Regions_Tool;
T_all_table(2:(n_t_s+1),2:(n_t_r+1)) = num2cell(T_all_mat);

T_all_table((n_t_s+2),1) = {'Total per Target-Region'};
T_all_table((n_t_s+2),2:(n_t_r+1)) = num2cell(T_regions');

T_all_table(1,(n_t_r+2)) = {'Total per Target-Sector'};
T_all_table(2:(n_t_s+1),(n_t_r+2)) = num2cell(T_sectors);
T_all_table((n_t_s+2),(n_t_r+2)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_all_table),[datapath 'Target_Perspective_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% Single Final Supply Perpsective
if sum(ismember(index_results,3)) > 0;
FS_all(:,1) = FS_all_ind_year(:,k_ind,k_time);   
FS_all_mat = vec2mat(FS_all,163)';
FS_sectors = sum(FS_all_mat,2);
FS_regions = sum(FS_all_mat,1)';    
    
FS_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Final Supply Perspective: Impacts are allocated to the sectors (rows) and regions (columns) situated at the end of the supply chain (one step before final consumption)'});
FS_all_table(2:164,1) = Labels_Sectors_all_Tool;
FS_all_table(1,2:50) = Labels_Regions_all_Tool;
FS_all_table(2:164,2:50) = num2cell(FS_all_mat);

FS_all_table(165,1) = {'Total per Region of Final Supply'};
FS_all_table(165,2:50) = num2cell(FS_regions');

FS_all_table(1,51) = {'Total per Sector of Final Supply'};
FS_all_table(2:164,51) = num2cell(FS_sectors);
FS_all_table(165,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(FS_all_table),[datapath 'FinalSupply_Perspective_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Single Final Demand Perspective
if sum(ismember(index_results,4)) > 0;
FD_all(:,1) = FD_all_ind_year(:,k_ind,k_time);
FD_all_mat = vec2mat(FD_all,7)';
FD_cat = sum(FD_all_mat,2);
FD_regions = sum(FD_all_mat)';

FD_all_table(1,1) = strcat(indicator_name, {' - Year '}, num2str(year), {' - Final Demand Perspective: Impacts are allocated to the categories (rows) and regions (columns) of final demand'});
FD_all_table(2:8,1) = Labels_FinalDemandCategories;
FD_all_table(1,2:50) = Labels_Regions_all_Tool;
FD_all_table(2:8,2:50) = num2cell(FD_all_mat);

FD_all_table(9,1) = {'Total per Region of Final Demand'};
FD_all_table(9,2:50) = num2cell(FD_regions');

FD_all_table(1,51) = {'Total per Category of Final Demand'};
FD_all_table(2:8,51) = num2cell(FD_cat);
FD_all_table(9,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(FD_all_table),[datapath 'FinalDemand_Perspective_Year' int2str(year) '_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

end % end output in the unit of the indicator



%% Compile and save single perspectives as shares in total global impact
if sum(ismember(index_output,2)) > 0;    % output as global shares

mkdir(['' folder_name '/Results_in_Global_Shares/Single_Perspectives/Year_' int2str(year) '/' indicator_name '']);   

TOT = TOT_ind_year(k_ind,k_time) / Total_Global_Impacts(k_ind,k_time); 

% Single Production Perspective
if sum(ismember(index_results,[1])) > 0;
P_all(:,1) = P_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);
P_all_mat = vec2mat(P_all,163)';
P_sectors = sum(P_all_mat,2);
P_regions = sum(P_all_mat)';

P_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Production Perspective: Impacts are allocated to the sectors (rows) and regions (columns) where they are caused '});
P_all_table(2:164,1) = Labels_Sectors_all_Tool;
P_all_table(1,2:50) = Labels_Regions_all_Tool;
P_all_table(2:164,2:50) = num2cell(P_all_mat);

P_all_table(165,1) = {'Total per Region of Production'};
P_all_table(165,2:50) = num2cell(P_regions');

P_all_table(1,51) = {'Total per Sector of Production'};
P_all_table(2:164,51) = num2cell(P_sectors);
P_all_table(165,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(P_all_table),[datapath 'Production_Perspective_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% Single Target Perspective
if sum(ismember(index_results,2)) > 0;
T_all(:,1) = T_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
T_all_mat = vec2mat(T_all,n_t_s)';
T_sectors = sum(T_all_mat,2);
T_regions = sum(T_all_mat,1)';
    
T_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Target Perspective: Impacts are allocated to the target-sectors (rows) and target-regions (columns) that are finally supplied'});
T_all_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
T_all_table(1,2:(n_t_r+1)) = Labels_Target_Regions_Tool;
T_all_table(2:(n_t_s+1),2:(n_t_r+1)) = num2cell(T_all_mat);

T_all_table((n_t_s+2),1) = {'Total per Target-Region'};
T_all_table((n_t_s+2),2:(n_t_r+1)) = num2cell(T_regions');

T_all_table(1,(n_t_r+2)) = {'Total per Target-Sector'};
T_all_table(2:(n_t_s+1),(n_t_r+2)) = num2cell(T_sectors);
T_all_table((n_t_s+2),(n_t_r+2)) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(T_all_table),[datapath 'Target_Perspective_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

% Single Final Supply Perpsective
if sum(ismember(index_results,3)) > 0 
FS_all(:,1) = FS_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);  
FS_all_mat = vec2mat(FS_all,163)';
FS_sectors = sum(FS_all_mat,2);
FS_regions = sum(FS_all_mat,1)';    
    
FS_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Final Supply Perspective: Impacts are allocated to the sectors (rows) and regions (columns) situated at the end of the supply chain (one step before final consumption)'});
FS_all_table(2:164,1) = Labels_Sectors_all_Tool;
FS_all_table(1,2:50) = Labels_Regions_all_Tool;
FS_all_table(2:164,2:50) = num2cell(FS_all_mat);

FS_all_table(165,1) = {'Total per Region of Final Supply'};
FS_all_table(165,2:50) = num2cell(FS_regions');

FS_all_table(1,51) = {'Total per Sector of Final Supply'};
FS_all_table(2:164,51) = num2cell(FS_sectors);
FS_all_table(165,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(FS_all_table),[datapath 'FinalSupply_Perspective_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


% Single Final Demand Perspective
if sum(ismember(index_results,4)) > 0;
FD_all(:,1) = FD_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);
FD_all_mat = vec2mat(FD_all,7)';
FD_cat = sum(FD_all_mat,2);
FD_regions = sum(FD_all_mat)';

FD_all_table(1,1) = strcat(indicator_name, {' - Share in Total Global Impacts (in decimal numbers) - Year '}, num2str(year), {' - Final Demand Perspective: Impacts are allocated to the categories (rows) and regions (columns) of final demand'});
FD_all_table(2:8,1) = Labels_FinalDemandCategories;
FD_all_table(1,2:50) = Labels_Regions_all_Tool;
FD_all_table(2:8,2:50) = num2cell(FD_all_mat);

FD_all_table(9,1) = {'Total per Region of Final Demand'};
FD_all_table(9,2:50) = num2cell(FD_regions');

FD_all_table(1,51) = {'Total per Category of Final Demand'};
FD_all_table(2:8,51) = num2cell(FD_cat);
FD_all_table(9,51) = num2cell(TOT);

datapath = ['' folder_name '/Results_in_Global_Shares/Single_Perspectives/Year_' int2str(year) '/' indicator_name '/'];
writetable(table(FD_all_table),[datapath 'FinalDemand_Perspective_Year' int2str(year) '_' indicator_name '_GlobalShare.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


end % end output as global shares


end % end index indicator

end % end index years

end % end single perspectives









%% Compile and Save Multiple indicators as text files (one table for each year): 
if sum(ismember(index_results,[16:27])) > 0; 

% Procedure for each year    
k_time = 0;
for year = index_years;  
k_time = k_time + 1;

%% Compile and save linkages in the unit of the indicator
if sum(ismember(index_output,1)) > 0;    % output in the unit of the indicator

mkdir(['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '']); 

if sum(ismember(index_results,16)) > 0; 
Multiple_Indicators_P_all(1:7987,1:no_indicators) = P_all_ind_year(:,:,k_time);    
Multiple_Indicators_P_all_table(1,1) = strcat({'Region and Sector of Production (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_P_all_table(2:7988,1) = Labels_Production_Tool;
Multiple_Indicators_P_all_table(2:7988,2:(no_indicators+1)) = num2cell(Multiple_Indicators_P_all);
Multiple_Indicators_P_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_P_all_table),[datapath 'Production_all_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,17)) > 0;
Multiple_Indicators_T_all(1:n_t,1:no_indicators) = T_all_ind_year(:,:,k_time);    
Multiple_Indicators_T_all_table(1,1) = strcat({'Target-Sector-Regions (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_T_all_table(2:(n_t+1),1) = Labels_Target_Tool;
Multiple_Indicators_T_all_table(2:(n_t+1),2:(no_indicators+1)) = num2cell(Multiple_Indicators_T_all);
Multiple_Indicators_T_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_T_all_table),[datapath 'Target_all_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,18)) > 0; 
Multiple_Indicators_FS_all(1:7987,1:no_indicators) = FS_all_ind_year(:,:,k_time);     
Multiple_Indicators_FS_all_table(1,1) = strcat({'Region and Sector of Final Supply (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_FS_all_table(2:7988,1) = Labels_Production_Tool;
Multiple_Indicators_FS_all_table(2:7988,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FS_all);
Multiple_Indicators_FS_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators');
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FS_all_table),[datapath 'FinalSupply_all_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,19)) > 0; 
Multiple_Indicators_FD_all(1:343,1:no_indicators) = FD_all_ind_year(:,:,k_time);     
Multiple_Indicators_FD_all_table(1,1) = strcat({'Region and Category of Final Demand (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_FD_all_table(2:344,1) = Labels_FinalDemand_Tool;
Multiple_Indicators_FD_all_table(2:344,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FD_all);
Multiple_Indicators_FD_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators');
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FD_all_table),[datapath 'FinalDemand_all_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,20)) > 0; 
Multiple_Indicators_P_sectors(1:163,1:no_indicators) = P_sectors_ind_year(:,:,k_time);     
Multiple_Indicators_P_sectors_table(1,1) = strcat({'Sector of Production (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_P_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Multiple_Indicators_P_sectors_table(2:164,2:(no_indicators+1)) = num2cell(Multiple_Indicators_P_sectors);
Multiple_Indicators_P_sectors_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_P_sectors_table),[datapath 'Production_Sector_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,21)) > 0; 
Multiple_Indicators_T_sectors(1:n_t_s,1:no_indicators) = T_sectors_ind_year(:,:,k_time);       
Multiple_Indicators_T_sectors_table(1,1) = strcat({'Target-Sectors (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_T_sectors_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
Multiple_Indicators_T_sectors_table(2:(n_t_s+1),2:(no_indicators+1)) = num2cell(Multiple_Indicators_T_sectors);
Multiple_Indicators_T_sectors_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_T_sectors_table),[datapath 'Target_Sector_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,22)) > 0; 
Multiple_Indicators_FS_sectors(1:163,1:no_indicators) = FS_sectors_ind_year(:,:,k_time);       
Multiple_Indicators_FS_sectors_table(1,1) = strcat({'Sectors of Final Supply (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_FS_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Multiple_Indicators_FS_sectors_table(2:164,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FS_sectors);
Multiple_Indicators_FS_sectors_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FS_sectors_table),[datapath 'FinalSupply_Sector_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,23)) > 0; 
Multiple_Indicators_P_regions(1:49,1:no_indicators) = P_regions_ind_year(:,:,k_time);     
Multiple_Indicators_P_regions_table(1,1) = strcat({'Region of Production (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_P_regions_table(2:50,1) = Labels_Regions_all_Tool;
Multiple_Indicators_P_regions_table(2:50,2:(no_indicators+1)) = num2cell(Multiple_Indicators_P_regions);
Multiple_Indicators_P_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_P_regions_table),[datapath 'Production_Region_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,24)) > 0; 
Multiple_Indicators_T_regions(1:n_t_r,1:no_indicators) = T_regions_ind_year(:,:,k_time);     
Multiple_Indicators_T_regions_table(1,1) = strcat({'Target-Regions (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_T_regions_table(2:(n_t_r+1),1) = Labels_Regions_all_Tool(index_t_r);
Multiple_Indicators_T_regions_table(2:(n_t_r+1),2:(no_indicators+1)) = num2cell(Multiple_Indicators_T_regions);
Multiple_Indicators_T_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_T_regions_table),[datapath 'Target_Region_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,25)) > 0; 
Multiple_Indicators_FS_regions(1:49,1:no_indicators) = FS_regions_ind_year(:,:,k_time);     
Multiple_Indicators_FS_regions_table(1,1) = strcat({'Region of Final Supply (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_FS_regions_table(2:50,1) = Labels_Regions_all_Tool;
Multiple_Indicators_FS_regions_table(2:50,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FS_regions);
Multiple_Indicators_FS_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FS_regions_table),[datapath 'FinalSupply_Region_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,26)) > 0; 
Multiple_Indicators_FD_regions(1:49,1:no_indicators) = FD_regions_ind_year(:,:,k_time);     
Multiple_Indicators_FD_regions_table(1,1) = strcat({'Region of Final Demand (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_FD_regions_table(2:50,1) = Labels_Regions_all_Tool;
Multiple_Indicators_FD_regions_table(2:50,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FD_regions);
Multiple_Indicators_FD_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FD_regions_table),[datapath 'FinalDemand_Region_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,27)) > 0; 
Multiple_Indicators_FD_cat(1:7,1:no_indicators) = FD_cat_ind_year(:,:,k_time);     
Multiple_Indicators_FD_cat_table(1,1) = strcat({'Category of Final Demand (rows) for multiple Indicators (columns) - Year '}, num2str(year));
Multiple_Indicators_FD_cat_table(2:8,1) = Labels_FinalDemandCategories;
Multiple_Indicators_FD_cat_table(2:8,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FD_cat);
Multiple_Indicators_FD_cat_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FD_cat_table),[datapath 'FinalDemand_Category_Multiple_Indicators_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

end % End output in the unit of the indicator



%% Compile and save linkages as shares in total global impact
if sum(ismember(index_output,2)) > 0;    % output as global shares

mkdir(['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '']); 

if sum(ismember(index_results,16)) > 0; 
k_ind = 0;
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_P_all(1:7987,k_ind) = P_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end
Multiple_Indicators_P_all_table(1,1) = strcat({'Region and Sector of Production (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_P_all_table(2:7988,1) = Labels_Production_Tool;
Multiple_Indicators_P_all_table(2:7988,2:(no_indicators+1)) = num2cell(Multiple_Indicators_P_all);
Multiple_Indicators_P_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_P_all_table),[datapath 'Production_all_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,17)) > 0;
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_T_all(1:n_t,k_ind) = T_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end
Multiple_Indicators_T_all_table(1,1) = strcat({'Target-Sector-Regions (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_T_all_table(2:(n_t+1),1) = Labels_Target_Tool;
Multiple_Indicators_T_all_table(2:(n_t+1),2:(no_indicators+1)) = num2cell(Multiple_Indicators_T_all);
Multiple_Indicators_T_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_T_all_table),[datapath 'Target_all_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,18)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_FS_all(1:7987,k_ind) = FS_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end       
Multiple_Indicators_FS_all_table(1,1) = strcat({'Region and Sector of Final Supply (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_FS_all_table(2:7988,1) = Labels_Production_Tool;
Multiple_Indicators_FS_all_table(2:7988,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FS_all);
Multiple_Indicators_FS_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators');
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FS_all_table),[datapath 'FinalSupply_all_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,19)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_FD_all(1:343,k_ind) = FD_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end      
Multiple_Indicators_FD_all_table(1,1) = strcat({'Region and Category of Final Demand (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_FD_all_table(2:344,1) = Labels_FinalDemand_Tool;
Multiple_Indicators_FD_all_table(2:344,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FD_all);
Multiple_Indicators_FD_all_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators');
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FD_all_table),[datapath 'FinalDemand_all_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,20)) > 0;   
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_P_sectors(1:163,k_ind) = P_sectors_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end 
Multiple_Indicators_P_sectors_table(1,1) = strcat({'Sector of Production (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_P_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Multiple_Indicators_P_sectors_table(2:164,2:(no_indicators+1)) = num2cell(Multiple_Indicators_P_sectors);
Multiple_Indicators_P_sectors_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_P_sectors_table),[datapath 'Production_Sector_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,21)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_T_sectors(1:n_t_s,k_ind) = T_sectors_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Multiple_Indicators_T_sectors_table(1,1) = strcat({'Target-Sectors (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_T_sectors_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
Multiple_Indicators_T_sectors_table(2:(n_t_s+1),2:(no_indicators+1)) = num2cell(Multiple_Indicators_T_sectors);
Multiple_Indicators_T_sectors_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_T_sectors_table),[datapath 'Target_Sector_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,22)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_FS_sectors(1:163,k_ind) = FS_sectors_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Multiple_Indicators_FS_sectors_table(1,1) = strcat({'Sectors of Final Supply (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_FS_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Multiple_Indicators_FS_sectors_table(2:164,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FS_sectors);
Multiple_Indicators_FS_sectors_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FS_sectors_table),[datapath 'FinalSupply_Sector_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,23)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_P_regions(1:49,k_ind) = P_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end    
Multiple_Indicators_P_regions_table(1,1) = strcat({'Region of Production (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_P_regions_table(2:50,1) = Labels_Regions_all_Tool;
Multiple_Indicators_P_regions_table(2:50,2:(no_indicators+1)) = num2cell(Multiple_Indicators_P_regions);
Multiple_Indicators_P_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_P_regions_table),[datapath 'Production_Region_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,24)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_T_regions(1:n_t_r,k_ind) = T_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end    
Multiple_Indicators_T_regions_table(1,1) = strcat({'Target-Regions (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_T_regions_table(2:(n_t_r+1),1) = Labels_Regions_all_Tool(index_t_r);
Multiple_Indicators_T_regions_table(2:(n_t_r+1),2:(no_indicators+1)) = num2cell(Multiple_Indicators_T_regions);
Multiple_Indicators_T_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_T_regions_table),[datapath 'Target_Region_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,25)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_FS_regions(1:49,k_ind) = FS_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end 
Multiple_Indicators_FS_regions_table(1,1) = strcat({'Region of Final Supply (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_FS_regions_table(2:50,1) = Labels_Regions_all_Tool;
Multiple_Indicators_FS_regions_table(2:50,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FS_regions);
Multiple_Indicators_FS_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FS_regions_table),[datapath 'FinalSupply_Region_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,26)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_FD_regions(1:49,k_ind) = FD_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end    
Multiple_Indicators_FD_regions_table(1,1) = strcat({'Region of Final Demand (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_FD_regions_table(2:50,1) = Labels_Regions_all_Tool;
Multiple_Indicators_FD_regions_table(2:50,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FD_regions);
Multiple_Indicators_FD_regions_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FD_regions_table),[datapath 'FinalDemand_Region_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,27)) > 0; 
k_ind = 0;    
for i = index_indicators;
k_ind = k_ind + 1;    
Multiple_Indicators_FD_cat(1:7,k_ind) = FD_cat_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end      
Multiple_Indicators_FD_cat_table(1,1) = strcat({'Category of Final Demand (rows) for multiple Indicators (columns) as Global Shares (in decimal numbers) - Year '}, num2str(year));
Multiple_Indicators_FD_cat_table(2:8,1) = Labels_FinalDemandCategories;
Multiple_Indicators_FD_cat_table(2:8,2:(no_indicators+1)) = num2cell(Multiple_Indicators_FD_cat);
Multiple_Indicators_FD_cat_table(1,2:(no_indicators+1)) = Labels_Indicators(index_indicators)';
datapath = ['' folder_name '/Results_in_Global_Shares/Multiple_Indicators/Year_' int2str(year) '/'];
writetable(table(Multiple_Indicators_FD_cat_table),[datapath 'FinalDemand_Category_Multiple_Indicators_asGlobalShares_Year' int2str(year) '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end


end % End global shares


end % End timeline

end % End muliple indicators compiled





%% Compile and Save Timelines for single indicators as text files (one table for each indicator):

if sum(ismember(index_results,[28:39])) > 0; 
% Procedure for each year    
k_ind = 0;
for i = index_indicators;
k_ind = k_ind + 1;
indicator_name = char(Labels_Indicators(i));

%% Compile and save linkages in the unit of the indicator
if sum(ismember(index_output,1)) > 0;    % output in the unit of the indicator
    
mkdir(['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '']); 

if sum(ismember(index_results,28)) > 0; 
Timeline_P_all(1:7987,1:no_years) = P_all_ind_year(:,k_ind,:);    
Timeline_P_all_table(1,1) = strcat(indicator_name, {' - Region and Sector of Production (rows) as a timeline (columns)'});
Timeline_P_all_table(2:7988,1) = Labels_Production_Tool;
Timeline_P_all_table(2:7988,2:(no_years+1)) = num2cell(Timeline_P_all);
Timeline_P_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_P_all_table),[datapath 'Production_all_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,29)) > 0;
Timeline_T_all(1:n_t,1:no_years) = T_all_ind_year(:,k_ind,:);  
Timeline_T_all_table(1,1) = strcat(indicator_name, {' - Target-Sector-Regions (rows) as a timeline (columns)'});
Timeline_T_all_table(2:(n_t+1),1) = Labels_Target_Tool;
Timeline_T_all_table(2:(n_t+1),2:(no_years+1)) = num2cell(Timeline_T_all);
Timeline_T_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_T_all_table),[datapath 'Target_all_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,30)) > 0; 
Timeline_FS_all(1:7987,1:no_years) = FS_all_ind_year(:,k_ind,:);   
Timeline_FS_all_table(1,1) = strcat(indicator_name, {' - Region and Sector of Final Supply (rows) as a timeline (columns)'});
Timeline_FS_all_table(2:7988,1) = Labels_Production_Tool;
Timeline_FS_all_table(2:7988,2:(no_years+1)) = num2cell(Timeline_FS_all);
Timeline_FS_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_FS_all_table),[datapath 'FinalSupply_all_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,31)) > 0; 
Timeline_FD_all(1:343,1:no_years) = FD_all_ind_year(:,k_ind,:);
Timeline_FD_all_table(1,1) = strcat(indicator_name, {' - Region and Category of Final Demand (rows) as a timeline (columns)'});
Timeline_FD_all_table(2:344,1) = Labels_FinalDemand_Tool;
Timeline_FD_all_table(2:344,2:(no_years+1)) = num2cell(Timeline_FD_all);
Timeline_FD_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_FD_all_table),[datapath 'FinalDemand_all_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,32)) > 0; 
Timeline_P_sectors(1:163,1:no_years) = P_sectors_ind_year(:,k_ind,:);    
Timeline_P_sectors_table(1,1) = strcat(indicator_name, {' - Sector of Production (rows) as a timeline (columns)'});
Timeline_P_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Timeline_P_sectors_table(2:164,2:(no_years+1)) = num2cell(Timeline_P_sectors);
Timeline_P_sectors_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_P_sectors_table),[datapath 'Production_Sector_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,33)) > 0; 
Timeline_T_sectors(1:n_t_s,1:no_years) = T_sectors_ind_year(:,k_ind,:); 
Timeline_T_sectors_table(1,1) = strcat(indicator_name, {' - Target-Sectors (rows) as a timeline (columns)'});
Timeline_T_sectors_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
Timeline_T_sectors_table(2:(n_t_s+1),2:(no_years+1)) = num2cell(Timeline_T_sectors);
Timeline_T_sectors_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_T_sectors_table),[datapath 'Target_Sector_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,34)) > 0; 
Timeline_FS_sectors(1:163,1:no_years) = FS_sectors_ind_year(:,k_ind,:);  
Timeline_FS_sectors_table(1,1) = strcat(indicator_name, {' - Sectors of Final Supply (rows) as a timeline (columns)'});
Timeline_FS_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Timeline_FS_sectors_table(2:164,2:(no_years+1)) = num2cell(Timeline_FS_sectors);
Timeline_FS_sectors_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_FS_sectors_table),[datapath 'FinalSupply_Sector_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,35)) > 0; 
Timeline_P_regions(1:49,1:no_years) = P_regions_ind_year(:,k_ind,:);      
Timeline_P_regions_table(1,1) = strcat(indicator_name, {' - Region of Production (rows) as a timeline (columns)'});
Timeline_P_regions_table(2:50,1) = Labels_Regions_all_Tool;
Timeline_P_regions_table(2:50,2:(no_years+1)) = num2cell(Timeline_P_regions);
Timeline_P_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_P_regions_table),[datapath 'Production_Region_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,36)) > 0; 
Timeline_T_regions(1:n_t_r,1:no_years) = T_regions_ind_year(:,k_ind,:);   
Timeline_T_regions_table(1,1) = strcat(indicator_name, {' - Target-Regions (rows) as a timeline (columns)'});
Timeline_T_regions_table(2:(n_t_r+1),1) = Labels_Regions_all_Tool(index_t_r);
Timeline_T_regions_table(2:(n_t_r+1),2:(no_years+1)) = num2cell(Timeline_T_regions);
Timeline_T_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_T_regions_table),[datapath 'Target_Region_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,37)) > 0; 
Timeline_FS_regions(1:49,1:no_years) = FS_regions_ind_year(:,k_ind,:);   
Timeline_FS_regions_table(1,1) = strcat(indicator_name, {' - Region of Final Supply (rows) as a timeline (columns)'});
Timeline_FS_regions_table(2:50,1) = Labels_Regions_all_Tool;
Timeline_FS_regions_table(2:50,2:(no_years+1)) = num2cell(Timeline_FS_regions);
Timeline_FS_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_FS_regions_table),[datapath 'FinalSupply_Region_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,38)) > 0; 
Timeline_FD_regions(1:49,1:no_years) = FD_regions_ind_year(:,k_ind,:);   
Timeline_FD_regions_table(1,1) = strcat(indicator_name, {' - Region of Final Demand (rows) as a timeline (columns)'});
Timeline_FD_regions_table(2:50,1) = Labels_Regions_all_Tool;
Timeline_FD_regions_table(2:50,2:(no_years+1)) = num2cell(Timeline_FD_regions);
Timeline_FD_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_FD_regions_table),[datapath 'FinalDemand_Region_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,39)) > 0; 
Timeline_FD_cat(1:7,1:no_years) = FD_cat_ind_year(:,k_ind,:);   
Timeline_FD_cat_table(1,1) = strcat(indicator_name, {' - Category of Final Demand (rows) as a timeline (columns)'});
Timeline_FD_cat_table(2:8,1) = Labels_FinalDemandCategories;
Timeline_FD_cat_table(2:8,2:(no_years+1)) = num2cell(Timeline_FD_cat);
Timeline_FD_cat_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/' indicator_name '/'];
writetable(table(Timeline_FD_cat_table),[datapath 'FinalDemand_Category_Timeline_' indicator_name '.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

end % end output in units




%% Compile and save linkages as shares in total global impact
if sum(ismember(index_output,2)) > 0;    % output as global shares

mkdir(['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '']); 

if sum(ismember(index_results,28)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_P_all(1:7987,k_time) = P_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end
Timeline_P_all_table(1,1) = strcat(indicator_name, {' - Region and Sector of Production (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_P_all_table(2:7988,1) = Labels_Production_Tool;
Timeline_P_all_table(2:7988,2:(no_years+1)) = num2cell(Timeline_P_all);
Timeline_P_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_P_all_table),[datapath 'Production_all_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,29)) > 0;
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_T_all(1:n_t,k_time) = T_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end    
Timeline_T_all_table(1,1) = strcat(indicator_name, {' - Target-Sector-Regions (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_T_all_table(2:(n_t+1),1) = Labels_Target_Tool;
Timeline_T_all_table(2:(n_t+1),2:(no_years+1)) = num2cell(Timeline_T_all);
Timeline_T_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_T_all_table),[datapath 'Target_all_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,30)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_FS_all(1:7987,k_time) = FS_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end          
Timeline_FS_all_table(1,1) = strcat(indicator_name, {' - Region and Sector of Final Supply (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_FS_all_table(2:7988,1) = Labels_Production_Tool;
Timeline_FS_all_table(2:7988,2:(no_years+1)) = num2cell(Timeline_FS_all);
Timeline_FS_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_FS_all_table),[datapath 'FinalSupply_all_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,31)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_FD_all(1:343,k_time) = FD_all_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end        
Timeline_FD_all_table(1,1) = strcat(indicator_name, {' - Region and Category of Final Demand (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_FD_all_table(2:344,1) = Labels_FinalDemand_Tool;
Timeline_FD_all_table(2:344,2:(no_years+1)) = num2cell(Timeline_FD_all);
Timeline_FD_all_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_FD_all_table),[datapath 'FinalDemand_all_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

if sum(ismember(index_results,32)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_P_sectors(1:163,k_time) = P_sectors_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end        
Timeline_P_sectors_table(1,1) = strcat(indicator_name, {' - Sector of Production (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_P_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Timeline_P_sectors_table(2:164,2:(no_years+1)) = num2cell(Timeline_P_sectors);
Timeline_P_sectors_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_P_sectors_table),[datapath 'Production_Sector_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,33)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_T_sectors(1:n_t_s,k_time) = T_sectors_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Timeline_T_sectors_table(1,1) = strcat(indicator_name, {' - Target-Sectors (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_T_sectors_table(2:(n_t_s+1),1) = Labels_Target_Sectors_Tool;
Timeline_T_sectors_table(2:(n_t_s+1),2:(no_years+1)) = num2cell(Timeline_T_sectors);
Timeline_T_sectors_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_T_sectors_table),[datapath 'Target_Sector_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,34)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_FS_sectors(1:163,k_time) = FS_sectors_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Timeline_FS_sectors_table(1,1) = strcat(indicator_name, {' - Sectors of Final Supply (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_FS_sectors_table(2:164,1) = Labels_Sectors_all_Tool;
Timeline_FS_sectors_table(2:164,2:(no_years+1)) = num2cell(Timeline_FS_sectors);
Timeline_FS_sectors_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_FS_sectors_table),[datapath 'FinalSupply_Sector_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,35)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_P_regions(1:49,k_time) = P_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end         
Timeline_P_regions_table(1,1) = strcat(indicator_name, {' - Region of Production (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_P_regions_table(2:50,1) = Labels_Regions_all_Tool;
Timeline_P_regions_table(2:50,2:(no_years+1)) = num2cell(Timeline_P_regions);
Timeline_P_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_P_regions_table),[datapath 'Production_Region_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,36)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_T_regions(1:n_t_r,k_time) = T_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Timeline_T_regions_table(1,1) = strcat(indicator_name, {' - Target-Regions (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_T_regions_table(2:(n_t_r+1),1) = Labels_Regions_all_Tool(index_t_r);
Timeline_T_regions_table(2:(n_t_r+1),2:(no_years+1)) = num2cell(Timeline_T_regions);
Timeline_T_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_T_regions_table),[datapath 'Target_Region_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,37)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_FS_regions(1:49,k_time) = FS_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Timeline_FS_regions_table(1,1) = strcat(indicator_name, {' - Region of Final Supply (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_FS_regions_table(2:50,1) = Labels_Regions_all_Tool;
Timeline_FS_regions_table(2:50,2:(no_years+1)) = num2cell(Timeline_FS_regions);
Timeline_FS_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_FS_regions_table),[datapath 'FinalSupply_Region_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,38)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_FD_regions(1:49,k_time) = FD_regions_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Timeline_FD_regions_table(1,1) = strcat(indicator_name, {' - Region of Final Demand (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_FD_regions_table(2:50,1) = Labels_Regions_all_Tool;
Timeline_FD_regions_table(2:50,2:(no_years+1)) = num2cell(Timeline_FD_regions);
Timeline_FD_regions_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_FD_regions_table),[datapath 'FinalDemand_Region_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

if sum(ismember(index_results,39)) > 0; 
k_time = 0;
for i = index_years;
k_time = k_time + 1;    
Timeline_FD_cat(1:7,k_time) = FD_cat_ind_year(:,k_ind,k_time) ./ Total_Global_Impacts(k_ind,k_time);    
end     
Timeline_FD_cat_table(1,1) = strcat(indicator_name, {' - Category of Final Demand (rows) as a timeline (columns) as shares in total global impacts (decimal numbers)'});
Timeline_FD_cat_table(2:8,1) = Labels_FinalDemandCategories;
Timeline_FD_cat_table(2:8,2:(no_years+1)) = num2cell(Timeline_FD_cat);
Timeline_FD_cat_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/' indicator_name '/'];
writetable(table(Timeline_FD_cat_table),[datapath 'FinalDemand_Category_Timeline_' indicator_name '_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end
    
end % End output as global shares 

end % End Indicators

end % End Timelines





%% Compile and Save Timelines for multiple Indictators as textfiles (total impacts for each indicator): 
if sum(ismember(index_results,40)) > 0;
    
%% Compile and save linkages in the unit of the indicator
if sum(ismember(index_output,1)) > 0;    % output in the unit of the indicator
mkdir(['' folder_name '/Results_in_Unit_of_Indicator/Timeline/']);  
Timeline_total_table(1,1) = {'Total scope 3 impacts of target-sector-regions for each indicator (rows) as a timeline (columns)'};
Timeline_total_table(2:(no_indicators+1),1) = Labels_Indicators(index_indicators);
Timeline_total_table(2:(no_indicators+1),2:(no_years+1)) = num2cell(TOT_ind_year);
Timeline_total_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Unit_of_Indicator/Timeline/'];
writetable(table(Timeline_total_table),[datapath 'Timeline_multiple_indicators_total.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end

%% Compile and save linkages as shares in total global impact
if sum(ismember(index_output,2)) > 0;    % output as global shares   
mkdir(['' folder_name '/Results_in_Global_Shares/Timeline/']); 
TOT_ind_year_shares = TOT_ind_year ./ Total_Global_Impacts;
Timeline_total_table(1,1) = {'Total scope 3 impacts of target-sector-regions for each indicator (rows) as a timeline (columns) - as a share in total global impacts (in decimal numbers)'};
Timeline_total_table(2:(no_indicators+1),1) = Labels_Indicators(index_indicators);
Timeline_total_table(2:(no_indicators+1),2:(no_years+1)) = num2cell(TOT_ind_year_shares);
Timeline_total_table(1,2:(no_years+1)) = num2cell(index_years);
datapath = ['' folder_name '/Results_in_Global_Shares/Timeline/'];
writetable(table(Timeline_total_table),[datapath 'Timeline_multiple_indicators_total_asGlobalShares.txt'], 'WriteVariableNames',false, 'delimiter', '\t');
end  

end

end % End main function