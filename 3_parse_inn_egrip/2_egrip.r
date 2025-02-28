library(RSelenium)
library(XML)
library(janitor) 
library(lubridate)
library(magrittr)
library(dplyr)
library(xlsx)

remDr <- remoteDriver(
    port = 4445L
)
remDr$open()

library("readxl")

goszak_df = read_excel('data-101113-2021-03-03-2.xlsx') #, header=TRUE, colClasses=NA)

goszak_df = goszak_df[352:702,]
goszak_df <- goszak_df[,1]
goszak_df[['ИНН']] <- rep('NA', dim(goszak_df)[1])
goszak_df[['name']] <- rep(0, dim(goszak_df)[1])
goszak_df[['text']] <- rep('NA', dim(goszak_df)[1])

library(stringdist)

get_name <- function(iii){
    name_box  <- remDr$findElement(using = "xpath", paste("/html/body/div[1]/div[3]/div/div[1]/div[4]/div[",toString(iii),"]/div[2]/a", sep=''))    
    name <-  name_box$getElementText()[[1]]
    return(name)
}

get_text <- function(iii){
    text_box  <- remDr$findElement(using = "xpath", paste("/html/body/div[1]/div[3]/div/div[1]/div[4]/div[",toString(iii),"]/div[3]/div", sep=''))
    text <-  text_box$getElementText()[[1]]
    return(text)
}

check_name <- function(iii, name){
    a1 <- gsub("[[:punct:]]", "", name)  # no libraries needed
    a2 <- gsub("[[:punct:]]", "", goszak_df[[iii,1]]) 
    check_name <- stringdist(tolower(a1),tolower(a2), method='lv')
    return(check_name == 0)
}

check <- TRUE
count <- 1
while (check==TRUE){
    remDr$navigate("https://egrul.nalog.ru/index.html")
    
    webElem <- remDr$findElement(using = "css", ".no-data")
    webElem$clickElement()
    
    Sys.sleep(1)
    
    webElem <- remDr$findElements("css", "iframe")
    remDr$switchToFrame(webElem[[1]])
    
    webElem <- remDr$findElement(using = "xpath", '//*[@id="chk_2"]')
    webElem$clickElement()
    
    webElem <- remDr$findElement(using = "xpath", '//*[@id="btn_ok"]')
    webElem$clickElement()
    
    windows <- remDr$getWindowHandles()
    remDr$switchToWindow(windows[[1]])
    
    webElem <- remDr$findElement(using = "css", "#query")
    webElem$sendKeysToElement(list(goszak_df[[count,1]], key = "enter"))
    
    Sys.sleep(4)
    
    box <- tryCatch({
        remDr$findElement(using = "xpath", "/html/body/div[1]/div[3]/div/div[1]/div[4]/div[1]/div[2]/a")
    }, error = function(warning_condition) {
        Sys.sleep(10)
        remDr$findElement(using = "xpath", "/html/body/div[1]/div[3]/div/div[1]/div[4]/div[1]/div[2]/a")
    })
    
    
    num_res <- 1 
    
    list_nums <-  c()
    
    while (num_res <= 20){
        name <- get_name(num_res)
        check_name_1 <- check_name(count,name)
        if ((length(list_nums)==0)&!check_name_1){
            num_res <- num_res+1
        } else if (check_name_1){
            list_nums <- c(list_nums, num_res) 
            num_res <- num_res+1
        } else if ((length(list_nums)!=0)&!check_name_1){
            num_res <- 21
        }
    }
    
    if (length(list_nums)==0){
#         check <- FALSE
        count = count + 1
        next
#         break
    }
    
    inn_list <- c()
    
    for (ij in list_nums){
        text <- get_text(ij)
        check_text <- grepl('прекращения', text, fixed = TRUE)
        if (!check_text){
            bb <- unlist(strsplit(text, ","))
            cc <- bb[grepl('ИНН', bb, fixed = TRUE)]
            inn <- unlist(strsplit(cc, ": "))[2]
            inn_list <- c(inn_list, inn)
        }
    }

    goszak_df[[count,3]] <- length(inn_list)   
    if (length(inn_list) == 1){
        goszak_df[[count,2]] <- inn_list
    } else {
        goszak_df[[count,2]] <- inn_list[1]
        goszak_df[[count,4]] <- paste(inn_list[-1], collapse = ', ')
    }
    
    
    count = count + 1

    if (count > dim(goszak_df)[1]){
        check <- FALSE
    }
    
    print(count)
    
    Sys.sleep(2)
}



write.xlsx(goszak_df, 'part_2.xlsx')