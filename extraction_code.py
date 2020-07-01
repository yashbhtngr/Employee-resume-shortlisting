import fitz
import nltk
import difflib as dfl
from pprint import pprint
from nltk.corpus import stopwords
from itertools import combinations 
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import re
import json

exp_wlist = ["experience","work history","employment history","professional experience","work experience", "professional history"]
skill_wlist = ["skills","professional skills","skillset","work skills","computer skills","added value","technical skills","pc skills","technical knowledge"]
edu_wlist = ["education","educational qualifications","qualifications","education and training"]
acc_wlist = ["accomplishments","achievements","professional accomplishments","professional achievements","awards", "awards and achievements"]
univ_sim = ["university","college","institution","academy","institute","université"]

nlp = en_core_web_sm.load()
stop = stopwords.words('english')

## Load the resume here
doc = fitz.open("resume1.pdf")
count = doc.pageCount
final_text =""
for i in range(count):
    page = doc[i]
    final_text = final_text + page.getText("text")
##

def n_seperator(target_list,otarget_list,text):
    required =""
    act_text = text.split("\n")
    attach=False
    other_sections=[]

    for x in otarget_list:
        other_sections=other_sections+x
    
    for line in act_text:
        words_in_line = line.split()
        words_in_line = [x.lower() for x in words_in_line]
        
        if len(words_in_line) <= 4:
            best_match = dfl.get_close_matches(line.lower(),target_list,1)
            other_bmatch = dfl.get_close_matches(line.lower(),other_sections,1)
            if len(best_match)!=0:
                attach=True
                required=required+line+"\n"
            elif len(other_bmatch)!=0:
                attach=False
            elif attach:
                required=required+line+"\n"
        elif attach:
            required=required+line+"\n"
    return required

def contains(word):
    found=False
    for w in univ_sim:
        nword = word.lower()
        if w in nword:
            found = True
            break
    return found

def find_univ(ent_list):
    all_ins=[]
    for ents in ent_list:
        found = contains(ents)
        if found:
            all_ins.append(ents)
    return all_ins

def sublists(l):
    if len(l)==0:
        return []
    else:
        rest = l[1:]
        final_l = []
        for i in range(1,len(l)+1):
            final_l = final_l + [l[0:i]]
        subsub = sublists(rest)
        return final_l+ subsub

def get_tokens(text):
    text_sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(x) for x in text_sentences]
    return tokenized_sentences

def check_cluster(app_skills,wanted_skills):
    distinct_clusters = [skills_dict[x] for x in wanted_skills]
    distinct_clusters = list(set(distinct_clusters))
    app_clusters = [skills_dict[x] for x in app_skills]
    c_score=0
    for x in app_clusters:
        if x in distinct_clusters:
            c_score = c_score + 1
    return c_score 


ph_numbers = re.findall(" *[0-9]{3}-*[0-9]{3}-*[0-9]{4} *", final_text)
phone_numbers = [x.strip() for x in ph_numbers]
phone_numbers = list(set(phone_numbers))
emails = re.findall("\S+@\S+", final_text)
emails = list(set(potential_emails))

exp_text = n_seperator(exp_wlist,[edu_wlist,skill_wlist,acc_wlist],final_text)
ed_text = n_seperator(edu_wlist,[exp_wlist,skill_wlist,acc_wlist],final_text)
skills_text = n_seperator(skill_wlist,[exp_wlist,edu_wlist,acc_wlist],final_text)
acc_text = n_seperator(acc_wlist,[exp_wlist,skill_wlist,edu_wlist],final_text)

ed_sentences = nltk.sent_tokenize(ed_text)
all_univs=[]

for sent in ed_sentences:
    sent=sent.lstrip()
    sent  = ''.join([x for x in sent if x != '\n'])
    ed_doc = nlp(sent)
    orgs_list = [x.text for x in ed_doc.ents if x.label_ == 'ORG']
    univs = find_univ(orgs_list)
    all_univs=all_univs+univs
all_univs = list(set(all_univs))

univ_file = open('top_univ.txt','r')
top_univs = (univ_file.read()).split('\n')
univ_score = 0
univ_matches=[]
for x in all_univs:
    close_matches=dfl.get_close_matches(x,top_univs,1,0.8)
    if len(close_matches)!=0:
        if close_matches[0] not in univ_matches:
            univ_score=univ_score+1
            univ_matches.append(close_matches[0])





# Initialize the skills dictionary (skills dictionary contains the skills->cluster_id mapping) here
skills_dict = {}
#

# use skills text which has been extracted above to get the skills of the applicant

applicant_skills = []
#

# FROM HIRING COMPANY SIDE
skills_wanted = []
#

common_skills= set(skills_wanted) & set(applicant_skills)
exact_match_score=2*(len(common_skills))
left_skills = list(set(applicant_skills)-common_skills)
rest_score = check_cluster(left_skills,skills_wanted)
skill_score = rest_score+exact_match_score

## The follwoing is the database
company_names = {"EXPERIAN CORP","CANON, INC"} # for the time being has only two company names 
## Have to initialize this *|*
hyper_parameter1 = 9
exp_sentences = exp_text.split('\n')
exp_sentences= [x.strip() for x in exp_sentences]
exp_sentences = [x for x in exp_sentences if x!='']
comp_job_sections =[]
attach=False
app_companies=[]
current_company=""
current_job=""
section =""
company_date=""
current_usedate=""
new_usedate=""
siter=0
for sent in exp_sentences:
    words = sent.split()
    last_company=""
    if len(words) < hyper_parameter1:
        n_grams = sublists(words)
        last_company=""
        for name in n_grams:
            name=' '.join(name)
            if name in company_names:
                last_company = name
        if last_company!="":
            app_companies.append(last_company)
            check_text = ' '.join(exp_sentences[siter:siter+2])
            dates = get_date(check_text)
            if len(dates)!=0:
                company_date = dates[0]
            
    jobs = [x[2] for x in finder.findall(sent)]
    if len(jobs)==0:
        if last_company!="" and attach==True:
            comp_job_sections.append((current_company,current_job,section,current_usedate))
            section=""
            current_company = last_company
            current_job=""
            attach=False
        elif last_company!="" and attach==False:
            current_company= last_company
            current_job=""
            attach=False
        elif last_company=="" and attach==True:
            section= section+'\n'+sent
    else:
        check_text = ' '.join(exp_sentences[siter:siter+2])
        dates = get_date(check_text)
        if len(dates)==0 and company_date=="" and attach==True:
            section=section+'\n'+sent
        elif len(dates)!=0 or company_date!="":
            
            if len(dates)!=0:
                new_usedate = dates[0]
            else:
                new_usedate = company_date
            if last_company!="" and attach==True:
                comp_job_sections.append((current_company,current_job,section,current_usedate))
                section=""
                current_company = last_company
                current_job=jobs[0]
                current_usedate = new_usedate
                new_usedate=""
                attach=True
            elif last_company!="" and attach==False:
                current_company= last_company
                current_job=jobs[0]
                current_usedate = new_usedate
                new_usedate=""
                attach=True
            elif last_company=="" and attach==False:
                current_job=jobs[0]
                current_usedate = new_usedate
                new_usedate=""
                attach=True
            elif last_company=="" and attach==True:
                comp_job_sections.append((current_company,current_job,section,current_usedate))
                current_job=jobs[0]
                section=""
                current_usedate = new_usedate
                new_usedate=""
        
    siter=siter+1

if section!="":
    comp_job_sections.append((current_company,current_job,section,current_usedate))

if len(comp_job_sections)!=0:
    first_date = comp_job_sections[0][3]
    last_date = comp_job_sections[-1][3]
    fd_fp = (re.findall("\W*[0-9]{4}\W*",first_date))[0]
    ld_fp = (re.findall("\W*[0-9]{4}\W*",last_date))[0]
    if fd_fp> ld_fp:
        comp_job_sections.reverse()




## comp_job_sections contains tuples of (company_name,job_name,job_data,tenure) job data will be used to create job_profile vectors
## the tuples are in ascending order of time
## So here is where you will create the job profile vectors and the company vectors