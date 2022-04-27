#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
struct Books {
   char  title[50];
   char  author[50];
   char  subject[100];
   int   book_id;
};

class storage{
    public:
    char  title[50];
    char  author[50];
    char  subject[100];
    int   book_id;

    storage(){
         strcpy(title, "");
        strcpy(author, "");
        strcpy(subject, "");
        book_id = 0;       
    }

    storage(char* t, char* a, char* sub, int id)
    {
        strcpy(title, t);
        strcpy(author, a);
        strcpy(subject, sub);
        book_id = id;
    }

    void display(){
        std::cout<<"Test method"<<std::endl;

        std::cout<< "title :" << title << std::endl;
        std::cout<< "author :" << author << std::endl;
        std::cout<< "subject :" << subject << std::endl;
        std::cout<< "book_id :" << book_id << std::endl;
    }

};


int main(){    
    
    int test = 0; 
    // test: 0 for struct, 1 for class, 2 for STL, 3 for JSON
    switch(test){

        case 0: //Test Struct
        {
            struct Books Book1;

            strcpy( Book1.title, "Learn C++ Programming");
            strcpy( Book1.author, "Chand Miyan"); 
            strcpy( Book1.subject, "C++ Programming");
            Book1.book_id = 6495407;

            char* test_buffer = (char*) new Books;

            std::cout<< sizeof(Book1)<<std::endl;

            memcpy(test_buffer, &Book1, sizeof(Book1));
            
            //creating ofstream class object

            std::ofstream fout;

            fout.open("datas.dat", std::ios::out | std::ios::binary);

            if (!fout){
                std::cout<< "Cannot open file." <<std::endl;
                return 1;
            }
            fout.write(test_buffer, sizeof(Books));
            fout.close();

            std::ifstream fin;

            fin.open("datas.dat", std::ios::out | std::ios::binary);

            if (!fin){
                std::cout<<"File cannot be opened."<<std::endl;
            }

            Books book2;
            fin.read((char*) &book2, sizeof(Books));
            fin.close();

            //see if the read is successful

            std::cout<< "title :" << book2.title << std::endl;
            std::cout<< "author :" << book2.author << std::endl;
            std::cout<< "subject :" << book2.subject << std::endl;
            std::cout<< "book_id :" << book2.book_id << std::endl;

            delete test_buffer;
        }
        break;

        case 1: //Test Class
        {
            char title[]= "Learn C++ Programming";
            char author[]= "Chand Miyan";
            char subject[]= "C++ Programming";
            int   book_id = 5;
            storage book2(title, author, subject, book_id);

            char* test_buffer = (char*) new storage;

            memcpy(test_buffer, &book2, sizeof(book2));

            std::ofstream fout;

            fout.open("data_c.dat", std::ios::out | std::ios::binary);

            if (!fout){
                std::cout<< "Cannot open file." <<std::endl;
                return 1;
            }
            fout.write(test_buffer, sizeof(Books));
            fout.close();

            std::ifstream fin;

            fin.open("data_c.dat", std::ios::out | std::ios::binary);

            if (!fin){
                std::cout<<"File cannot be opened."<<std::endl;
            }

            storage book3;
            fin.read((char*) &book3, sizeof(storage));
            fin.close();            
            book3.display();

            delete test_buffer; 

        }
        break;

        case 2: //STL vectors
        /*
        {
            std::vector <int> t11; 

            for (int i=1; i<=100; i++){
                t11.push_back(i);
            }

            std::cout<< "vector data: " << t11.size() <<std::endl;
            auto data = t11.data();
            char* test_buffer = (char*) malloc(t11.size()*sizeof(int));

            memcpy(test_buffer, data, t11.size()*sizeof(int));
            
            //write to a file
            std::ofstream fout;

            fout.open("data_v.dat", std::ios::out | std::ios::binary);

            if (!fout){
                std::cout<< "Cannot open file." <<std::endl;
                return 1;
            }
            fout.write(test_buffer, t11.size());
            fout.close();

            std::ifstream fin;

            fin.open("data_v.dat", std::ios::out | std::ios::binary);

            if (!fin){
                std::cout<<"File cannot be opened."<<std::endl;
            }

            std::vector<int> t12;
            char* data_receive = new (t11.size()*sizeof(int))
            //fin.read((char*) data_receive, t11.size()*sizeof(int));
            fin.close();

            //print the vector
            std::cout<<"Reading from saved file"<<std::endl;

            for (auto i= data_receive; i<=(data_receive+100); i++){
                std::cout<<*i;
            }            

            // STL vectors cannot be copied using this method as the data structure uses pointers to reference the data elements
            delete data_receive;
            free(test_buffer);
        }
        */
        break;

        case 3: //JSON
        {

        }
        break;

    }



    return 0;
}