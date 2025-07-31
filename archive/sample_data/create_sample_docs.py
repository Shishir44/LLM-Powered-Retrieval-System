#!/usr/bin/env python3
"""
Script to create comprehensive sample customer support documents
for testing the knowledge base and conversation services.
"""

import os
import json

def create_documents():
    base_path = '/Users/shishirkafle/Desktop/ChatBoq/LLM-Powered-Retrieval-System/sample_support_documents'
    
    # Additional documents to create
    additional_docs = [
        # Product Information
        {
            'category': 'product_info',
            'filename': 'accessories_catalog.txt',
            'content': '''ACCESSORIES CATALOG

HEADPHONES AND AUDIO

Sony WH-1000XM5 Wireless Headphones:
- Price: $399.99
- Features: Industry-leading noise cancellation, 30-hour battery
- Colors: Black, Silver
- Connectivity: Bluetooth 5.2, 3.5mm wired
- Weight: 250g
- Warranty: 1 year manufacturer warranty

Apple AirPods Pro (2nd generation):
- Price: $249.99
- Features: Active Noise Cancellation, Spatial Audio
- Battery: 6 hours listening, 30 hours with case
- Charging: MagSafe, Lightning, Qi wireless
- Colors: White only
- Warranty: 1 year limited warranty

CHARGERS AND CABLES

Anker PowerPort III 65W:
- Price: $55.99
- Features: GaN technology, foldable design
- Ports: 2x USB-C, 1x USB-A
- Compatibility: MacBook, iPhone, iPad, Android
- Safety: MultiProtect safety system
- Warranty: 18 months

Belkin Lightning to USB-C Cable:
- Price: $24.99
- Length: 3ft, 6ft, 10ft options
- Features: MFi certified, fast charging
- Compatibility: iPhone, iPad with Lightning
- Durability: 10,000+ bend tested
- Warranty: 2 year connected equipment warranty

CASES AND PROTECTION

OtterBox Defender Series iPhone Case:
- Price: $59.99 - $64.99
- Protection: Drop protection up to 4x military standard
- Features: Holster with belt clip, port covers
- Materials: Synthetic rubber and polycarbonate
- Colors: Black, Blue, Pink (varies by phone model)
- Warranty: Lifetime warranty against defects

UAG Pathfinder Laptop Sleeve:
- Price: $79.99 - $99.99
- Sizes: 13", 15", 16" laptop compatibility
- Features: Military-grade protection, water resistant
- Materials: Ballistic nylon exterior, fleece interior
- Colors: Black, Olive, Midnight Camo
- Warranty: 1 year manufacturer warranty

KEYBOARDS AND MICE

Logitech MX Master 3S Wireless Mouse:
- Price: $99.99
- Features: 8K DPI sensor, quiet clicks, electromagnetic scrolling
- Battery: 70 days on single charge
- Connectivity: Bluetooth, USB-C receiver
- Colors: Graphite, Pale Gray, Rose
- Warranty: 1 year limited hardware warranty

Keychron K3 Mechanical Keyboard:
- Price: $84.99 - $94.99
- Features: Low-profile mechanical switches, RGB backlight
- Layout: 75% compact layout, wireless/wired
- Switches: Gateron Low Profile (Red, Blue, Brown)
- Battery: 4000mAh, 240 hours usage
- Warranty: 1 year manufacturer warranty

STORAGE SOLUTIONS

Samsung T7 Portable SSD:
- Price: $89.99 (500GB) - $299.99 (2TB)
- Capacities: 500GB, 1TB, 2TB
- Speed: Up to 1,050 MB/s read/write
- Connectivity: USB 3.2 Gen 2, USB-C to USB-C and USB-A cables
- Security: Password protection, AES 256-bit encryption
- Warranty: 3 year limited warranty

WD My Passport External Hard Drive:
- Price: $54.99 (1TB) - $129.99 (5TB)
- Capacities: 1TB, 2TB, 4TB, 5TB
- Features: Automatic backup software, password protection
- Connectivity: USB 3.2 Gen 1, USB-A cable included
- Colors: Black, Blue, Red, White (availability varies)
- Warranty: 3 year limited warranty'''
        },
        
        # Policies
        {
            'category': 'policies',
            'filename': 'privacy_policy.md',
            'content': '''# Privacy Policy

## Information We Collect

### Personal Information
We collect information you provide directly to us, such as:
- **Account Information**: Name, email address, phone number, billing address
- **Purchase History**: Items purchased, payment methods, transaction details
- **Communication**: Customer service interactions, reviews, survey responses
- **Marketing Preferences**: Email subscriptions, promotional communications

### Automatically Collected Information
- **Device Information**: IP address, browser type, operating system
- **Usage Data**: Pages visited, time spent, clicks and navigation patterns
- **Location Data**: General geographic location based on IP address
- **Cookies**: Preferences, login status, shopping cart contents

## How We Use Your Information

### Order Processing
- Process and fulfill your orders
- Send order confirmations and shipping updates
- Handle returns and exchanges
- Provide customer support

### Account Management
- Create and maintain your account
- Authenticate your identity
- Provide personalized recommendations
- Remember your preferences and settings

### Marketing Communications
- Send promotional emails (with your consent)
- Notify you about sales and special offers
- Conduct customer satisfaction surveys
- Provide product updates and announcements

### Website Improvement
- Analyze website usage and performance
- Troubleshoot technical issues
- Enhance user experience
- Develop new features and services

## Information Sharing

### Third-Party Service Providers
We share information with trusted partners who help us:
- **Payment Processing**: Secure handling of payment transactions
- **Shipping**: Delivery of your orders
- **Customer Service**: Support ticket management and live chat
- **Marketing**: Email campaigns and advertising platforms
- **Analytics**: Website performance and user behavior analysis

### Legal Requirements
We may disclose information when required by law or to:
- Comply with legal processes or government requests
- Protect our rights, property, or safety
- Prevent fraud or security threats
- Enforce our terms of service

### Business Transfers
In the event of a merger, acquisition, or sale of assets, customer information may be transferred to the new entity.

## Data Security

### Protection Measures
- **Encryption**: All sensitive data encrypted in transit and at rest
- **Access Controls**: Strict employee access on need-to-know basis
- **Monitoring**: Continuous security monitoring and threat detection
- **Updates**: Regular security patches and system updates

### Payment Security
- **PCI Compliance**: Meet Payment Card Industry security standards
- **Tokenization**: Credit card numbers replaced with secure tokens
- **No Storage**: We don't store complete credit card information
- **Fraud Detection**: Advanced systems monitor for suspicious activity

## Your Rights and Choices

### Account Access
- **View Information**: Access your personal data in your account
- **Update Details**: Modify your information at any time
- **Delete Account**: Request account deletion (subject to legal requirements)
- **Download Data**: Request a copy of your personal information

### Marketing Communications
- **Unsubscribe**: Opt out of promotional emails via unsubscribe link
- **Preferences**: Update communication preferences in your account
- **Do Not Sell**: Opt out of personal information sales (where applicable)

### Cookies and Tracking
- **Browser Settings**: Control cookies through browser preferences
- **Opt-Out Tools**: Use industry opt-out tools for advertising
- **Do Not Track**: We honor Do Not Track browser signals

## International Users

### Data Transfers
For international customers, your information may be transferred to and processed in the United States or other countries where our service providers operate.

### GDPR Rights (EU Residents)
Under the General Data Protection Regulation, you have the right to:
- **Access**: Request a copy of your personal data
- **Rectification**: Correct inaccurate information
- **Erasure**: Request deletion of your data
- **Portability**: Receive your data in a machine-readable format
- **Objection**: Object to processing for marketing purposes
- **Restriction**: Limit how we process your information

## Children's Privacy

We do not knowingly collect personal information from children under 13. If we discover we have collected information from a child under 13, we will delete it immediately.

## Changes to This Policy

We may update this privacy policy periodically. We will notify you of significant changes by:
- Posting the updated policy on our website
- Sending email notifications to registered users
- Displaying prominent notices on our website

## Contact Information

For privacy-related questions or concerns:
- **Email**: privacy@company.com
- **Phone**: 1-800-555-0132
- **Mail**: Privacy Officer, 123 Main Street, Anytown, ST 12345

**Data Protection Officer** (for EU inquiries):
- **Email**: dpo@company.com

Last updated: January 1, 2024'''
        },
        
        # Troubleshooting
        {
            'category': 'troubleshooting',
            'filename': 'software_installation.txt',
            'content': '''SOFTWARE INSTALLATION TROUBLESHOOTING

GENERAL INSTALLATION ISSUES

Insufficient Privileges:
- Run installer as Administrator (Windows) or with sudo (Mac/Linux)
- Right-click installer and select "Run as administrator"
- Disable User Account Control temporarily if needed
- Ensure user account has software installation permissions
- Contact system administrator for corporate computers

Insufficient Disk Space:
- Check available storage before installation
- Clean temporary files and downloads folder
- Uninstall unused programs to free space
- Move large files to external storage or cloud
- Consider installing to different drive with more space

Corrupt Download:
- Re-download installer from official source
- Verify file size matches expected size
- Check file hash/checksum if provided
- Download using different browser or network
- Clear browser cache before downloading

WINDOWS-SPECIFIC ISSUES

Windows Installer Service Issues:
- Restart Windows Installer service in Services.msc
- Run Windows Installer Cleanup Utility
- Register Windows Installer service: msiexec /unregister then msiexec /regserver
- Check Windows Update for installer updates
- Run System File Checker: sfc /scannow

.NET Framework Requirements:
- Install required .NET Framework version
- Download from Microsoft official website
- Restart computer after .NET installation
- Check if newer version is already installed
- Use .NET Framework Repair Tool if corrupted

Visual C++ Redistributables:
- Install Microsoft Visual C++ Redistributable packages
- Download both x86 and x64 versions if unsure
- Install all years (2015-2022) for maximum compatibility
- Restart computer after installation
- Check Windows Update for latest versions

MAC-SPECIFIC ISSUES

Gatekeeper Blocking Installation:
- Control-click installer and select "Open"
- Go to System Preferences > Security & Privacy > General
- Click "Open Anyway" for blocked application
- Disable Gatekeeper temporarily: sudo spctl --master-disable
- Re-enable after installation: sudo spctl --master-enable

Insufficient Permissions:
- Use sudo command for command-line installations
- Check if installer requires admin password
- Verify user account has admin privileges
- Try installing to user Applications folder instead of system folder
- Repair disk permissions using Disk Utility

Code Signing Issues:
- Verify application is from trusted developer
- Check developer certificate validity
- Use command: codesign --verify --verbose /path/to/app
- Clear system caches if signature verification fails
- Contact developer if signature is invalid

MOBILE APP INSTALLATION

iOS App Store Issues:
- Check available storage space on device
- Verify Apple ID payment information is current
- Sign out and back into App Store
- Reset network settings if download stalls
- Check if app is compatible with iOS version

Android Play Store Issues:
- Clear Google Play Store cache and data
- Check Google account payment methods
- Verify device compatibility with app requirements
- Enable installation from unknown sources for sideloading
- Check if device has sufficient RAM for app

ANTIVIRUS INTERFERENCE

False Positive Detection:
- Temporarily disable real-time protection
- Add installer to antivirus exclusions
- Download installer again after exclusion setup
- Submit false positive report to antivirus vendor
- Use different antivirus if persistent issues

Quarantine Recovery:
- Access antivirus quarantine/vault
- Restore installer file from quarantine
- Add installer to trusted files list
- Disable heuristic scanning temporarily
- Whitelist installer publisher/developer

DRIVER INSTALLATION ISSUES

Hardware Not Detected:
- Check Device Manager for unknown devices
- Verify hardware is properly connected
- Try different USB port or cable
- Download drivers before connecting hardware
- Install drivers in safe mode if normal mode fails

Driver Signature Issues:
- Disable driver signature enforcement temporarily
- Use certified drivers from manufacturer website
- Install drivers in compatibility mode
- Use Windows Update to find certified drivers
- Contact manufacturer for signed drivers

Legacy Hardware:
- Check manufacturer website for legacy driver support
- Use compatibility mode for older drivers
- Consider third-party driver solutions carefully
- Verify hardware is supported on current OS version
- Look for generic drivers that might work

SOFTWARE-SPECIFIC TROUBLESHOOTING

Adobe Products:
- Use Adobe Creative Cloud Cleaner Tool
- Install with internet connection for license verification
- Check system requirements carefully
- Disable third-party security software during installation
- Install as administrator with all users permission

Microsoft Office:
- Use Microsoft Support and Recovery Assistant
- Completely uninstall previous versions first
- Install with account that has admin privileges
- Disable antivirus during installation
- Use offline installer for connection issues

Browsers:
- Uninstall existing version completely
- Clear all user data if fresh install desired
- Install with administrator privileges
- Check for conflicting browser extensions or toolbars
- Install as standard user after initial admin install

INSTALLATION BEST PRACTICES

Pre-Installation Checklist:
- Create system restore point
- Close all running applications
- Disable antivirus temporarily
- Check system requirements
- Ensure stable internet connection for online installers

During Installation:
- Don't interrupt installation process
- Don't use computer for other tasks during install
- Watch for custom install options vs express install
- Read license agreements for important terms
- Note installation location for future reference

Post-Installation:
- Restart computer if prompted
- Re-enable antivirus software
- Run software to verify successful installation
- Register software if required
- Check for software updates immediately

GETTING HELP

When to Contact Support:
- Installation fails repeatedly after troubleshooting
- Error messages that don't match common issues
- Hardware driver installation problems
- Corporate software deployment issues
- License or activation problems

Information to Provide:
- Exact error messages (screenshots helpful)
- Operating system version and architecture
- Hardware specifications
- Previous troubleshooting attempts
- Installation log files if available

Contact Information:
Phone: 1-800-555-0133 (Software Support)
Email: software@company.com
Chat: Available 24/7 for installation issues
Remote Support: Available for complex installation problems'''
        },

        # More FAQs
        {
            'category': 'faqs',
            'filename': 'billing_faq.md',
            'content': '''# Billing and Payment FAQ

## Payment Methods

### Q: What payment methods do you accept?
**A:** We accept:
- **Credit Cards**: Visa, MasterCard, American Express, Discover
- **Debit Cards**: All major debit cards with Visa/MasterCard logo
- **Digital Wallets**: PayPal, Apple Pay, Google Pay, Samsung Pay
- **Buy Now, Pay Later**: Klarna, Affirm (subject to approval)
- **Gift Cards**: Our store gift cards and e-gift cards
- **Corporate**: Purchase orders for approved business accounts

### Q: Is it safe to enter my credit card information?
**A:** Absolutely. We use industry-standard security measures:
- **256-bit SSL encryption** for all transactions
- **PCI DSS Level 1 compliance** for payment processing
- **Tokenization** - we don't store your complete card details
- **Fraud monitoring** on all transactions
- **Two-factor authentication** for suspicious activity

### Q: Can I save multiple payment methods?
**A:** Yes, you can save multiple payment methods in your account:
- Add up to 5 credit/debit cards
- Link PayPal and other digital wallets
- Set a default payment method for faster checkout
- Update or remove saved methods anytime
- All saved information is encrypted and secure

## Billing and Invoices

### Q: When will I be charged for my order?
**A:** Billing timeline depends on the product type:
- **Digital products**: Charged immediately upon purchase
- **In-stock items**: Charged when your order ships
- **Pre-orders**: Charged when the item becomes available
- **Custom orders**: May require deposit, final charge when shipped

### Q: How do I get a receipt or invoice?
**A:** Receipts are automatically provided:
- **Email receipt** sent immediately after purchase
- **PDF attachment** included in confirmation email
- **Account history** - view all receipts in your account
- **Resend receipt** - request duplicate via customer service
- **Business invoices** available for corporate accounts

### Q: Can I get a refund to a different payment method?
**A:** Refund policy for payment methods:
- **Same method preferred** - refunds typically go to original payment
- **Store credit option** - available if original method unavailable
- **Different card** - possible in special circumstances
- **Cash refunds** - available for in-store purchases under $50
- **Processing time** - 3-7 business days depending on method

## Subscription and Recurring Billing

### Q: How do subscription renewals work?
**A:** Subscription billing process:
- **Automatic renewal** unless cancelled before next billing date
- **Email reminder** sent 7 days before renewal
- **Charge date** - same date each month/year as original purchase
- **Price changes** - 30 days notice for subscription price increases
- **Failed payment** - 3 retry attempts over 7 days

### Q: How do I cancel my subscription?
**A:** Cancellation options:
- **Online** - Cancel in your account settings anytime
- **Phone** - Call customer service during business hours
- **Email** - Send cancellation request to billing@company.com
- **Effective date** - Cancellation takes effect at end of current billing period
- **Confirmation** - Cancellation confirmation sent via email

### Q: Can I get a partial refund if I cancel mid-cycle?
**A:** Subscription refund policy:
- **No partial refunds** for monthly subscriptions
- **Pro-rated refunds** may be available for annual subscriptions
- **Service continues** until end of paid period after cancellation
- **Exceptions** - refunds considered for technical issues or billing errors
- **Contact support** for specific refund requests

## International Billing

### Q: Do you accept international credit cards?
**A:** International payment acceptance:
- **Visa and MasterCard** accepted worldwide
- **American Express** accepted in most countries
- **Currency conversion** handled by your bank
- **Additional fees** may apply from your card issuer
- **Address verification** required for fraud prevention

### Q: What currencies do you accept?
**A:** Supported currencies:
- **Primary**: USD (US Dollar)
- **International**: CAD, EUR, GBP, JPY, AUD
- **Conversion rates** updated daily
- **Final amount** shown in your selected currency at checkout
- **Bank fees** for currency conversion may apply

### Q: Are taxes included in international orders?
**A:** Tax handling by region:
- **US orders** - Sales tax calculated based on shipping address
- **Canadian orders** - GST/HST added at checkout
- **EU orders** - VAT included in displayed prices
- **Other countries** - Duties and taxes may be collected by customs
- **Tax exemption** available for qualified organizations

## Billing Issues

### Q: My payment was declined. What should I do?
**A:** Common reasons and solutions:
- **Insufficient funds** - Check account balance
- **Incorrect information** - Verify billing address matches card
- **Security hold** - Contact your bank to authorize transaction
- **Daily limits** - Check if you've exceeded daily spending limit
- **International block** - Notify bank of international purchase

### Q: I was charged twice for the same order. How do I get a refund?
**A:** Duplicate charge resolution:
- **Check email** - Verify if two separate orders were placed
- **Temporary holds** - Some charges may be temporary authorizations
- **Contact us immediately** - Report duplicate charges within 5 days
- **Investigation time** - 3-5 business days to resolve
- **Automatic refund** - Issued once duplicate confirmed

### Q: I don't recognize a charge on my statement. What should I do?
**A:** Unrecognized charge steps:
1. **Check email** for order confirmations
2. **Review account** order history
3. **Family members** - Check if others used your card
4. **Merchant name** - We may appear as different name on statement
5. **Contact us** if charge remains unrecognized

## Business and Corporate Billing

### Q: Do you offer corporate accounts with net payment terms?
**A:** Corporate account features:
- **Net 30/60 terms** available for qualified businesses
- **Credit application** required for approval
- **Minimum order** requirements may apply
- **Dedicated rep** assigned to corporate accounts
- **Volume discounts** available for large orders

### Q: Can I get detailed invoices for expense reporting?
**A:** Business invoice options:
- **Itemized receipts** with product details and categories
- **PDF invoices** suitable for expense reports
- **Monthly statements** for corporate accounts
- **Tax ID numbers** included when provided
- **Custom fields** available for purchase orders

### Q: Do you provide W-9 forms or tax documents?
**A:** Tax documentation:
- **W-9 forms** available upon request
- **Annual summaries** for business customers
- **Sales tax certificates** processed for exempt organizations
- **International VAT** documentation for EU businesses
- **Contact accounting** department for specific tax needs

## Getting Help with Billing

### Q: Who should I contact for billing questions?
**A:** Billing support contacts:
- **Phone**: 1-800-555-0134 (Billing Department)
- **Email**: billing@company.com
- **Live Chat**: Available 24/7 for billing issues
- **Hours**: Monday-Friday 8 AM - 8 PM EST
- **Emergency**: After-hours support for payment failures

### Q: What information do I need when calling about billing?
**A:** Information to have ready:
- **Order number** or transaction ID
- **Last 4 digits** of payment method used
- **Billing address** associated with the order
- **Email address** used for the purchase
- **Description** of the billing issue
- **Screenshots** of error messages if applicable'''
        },
        
        # Additional troubleshooting
        {
            'category': 'troubleshooting',
            'filename': 'email_setup.txt',
            'content': '''EMAIL SETUP AND TROUBLESHOOTING

COMMON EMAIL CLIENTS

Microsoft Outlook Setup:
1. Open Outlook and click "File" > "Add Account"
2. Enter your email address and click "Connect"
3. Enter your password when prompted
4. If automatic setup fails, choose "Manual setup"
5. Select account type (IMAP recommended for most users)
6. Enter server settings provided by email provider
7. Test account settings and click "Done"

Apple Mail Setup:
1. Open System Preferences > Internet Accounts
2. Click "Add Account" and select email provider
3. Enter email address and password
4. Choose which services to sync (Mail, Contacts, Calendar)
5. Click "Done" to complete setup
6. For manual setup, choose "Other Mail Account"
7. Enter incoming and outgoing server details

Gmail Mobile App:
1. Download Gmail app from App Store or Play Store
2. Open app and tap "Add email address"
3. Select "Google" for Gmail accounts
4. Enter email and password, follow authentication steps
5. For other providers, select "Other"
6. Enter server settings manually if needed
7. Complete setup and verify email sync

EMAIL SERVER SETTINGS

IMAP Settings (Recommended):
- Allows access from multiple devices
- Emails stored on server
- Changes sync across all devices
- Better for users with multiple devices
- Uses more server storage

POP3 Settings:
- Downloads emails to single device
- Emails removed from server after download
- Good for single device access
- Uses less server storage
- Limited synchronization between devices

Common Provider Settings:
Gmail IMAP:
- Incoming: imap.gmail.com, Port 993, SSL
- Outgoing: smtp.gmail.com, Port 587, TLS
- Authentication required for both

Outlook/Hotmail:
- Incoming: outlook.office365.com, Port 993, SSL
- Outgoing: smtp-mail.outlook.com, Port 587, TLS
- Use email address as username

Yahoo Mail:
- Incoming: imap.mail.yahoo.com, Port 993, SSL
- Outgoing: smtp.mail.yahoo.com, Port 587, TLS
- May require app-specific password

COMMON EMAIL PROBLEMS

Cannot Send Emails:
- Check outgoing server settings (SMTP)
- Verify authentication is enabled
- Confirm correct username and password
- Check if ISP blocks outgoing email ports
- Try different SMTP port (587 instead of 25)
- Enable "Less secure app access" for some providers

Cannot Receive Emails:
- Verify incoming server settings (IMAP/POP3)
- Check internet connection
- Confirm email address spelling
- Look in spam/junk folder
- Check email quota/storage limits
- Verify account is active with provider

Authentication Errors:
- Double-check username and password
- Use full email address as username
- Enable two-factor authentication if required
- Generate app-specific password for secure accounts
- Check if account is locked or suspended
- Update email client to latest version

SSL/TLS Certificate Errors:
- Update email client software
- Check date and time settings on device
- Accept certificate temporarily to test
- Contact email provider about certificate issues
- Try disabling SSL temporarily (not recommended long-term)
- Clear email client cache and restart

MOBILE EMAIL ISSUES

iOS Mail Problems:
- Delete and re-add email account
- Check iOS version compatibility
- Verify cellular/WiFi connection
- Reset network settings if needed
- Check mail fetch settings (push vs fetch)
- Restart device to clear temporary issues

Android Email Issues:
- Clear Gmail app cache and data
- Check Google account sync settings
- Verify account permissions for email access
- Try different email app (Samsung Email, BlueMail)
- Check power saving modes affecting sync
- Ensure background app refresh is enabled

Sync Issues:
- Check sync frequency settings
- Verify account has proper permissions
- Look for sync errors in account settings
- Try manual refresh/sync
- Check available storage space
- Remove and re-add account if persistent

EMAIL SECURITY

Two-Factor Authentication:
- Enable 2FA on email account
- Use authenticator app instead of SMS when possible
- Generate app-specific passwords for email clients
- Keep backup codes in secure location
- Review and remove unused app passwords
- Monitor account for unauthorized access

Spam and Phishing Protection:
- Never click suspicious links in emails
- Verify sender identity before providing information
- Check email headers for authenticity
- Use built-in spam filters
- Report phishing attempts to provider
- Keep email client updated with security patches

Password Security:
- Use strong, unique passwords for email accounts
- Enable account recovery options (backup email, phone)
- Change passwords regularly
- Don't save passwords on shared computers
- Use password manager for complex passwords
- Monitor for data breach notifications

BUSINESS EMAIL SETUP

Exchange Server:
1. Obtain server details from IT administrator
2. Use Outlook for best compatibility
3. Enter server name, domain, username, password
4. Configure security settings as required
5. Test connection and sync settings
6. Setup mobile devices with same credentials

G Suite/Google Workspace:
1. Verify domain ownership with Google
2. Setup MX records with domain registrar
3. Create user accounts in admin console
4. Configure email clients with Gmail settings
5. Enable two-factor authentication
6. Setup mobile device management if required

Custom Domain Email:
- Configure MX records with hosting provider
- Setup email accounts in hosting control panel
- Use hosting provider's server settings
- Consider using email forwarding to major provider
- Backup email data regularly
- Monitor email deliverability

TROUBLESHOOTING STEPS

Basic Troubleshooting:
1. Restart email application
2. Check internet connection
3. Verify server settings
4. Test with webmail interface
5. Clear application cache
6. Update email client software

Advanced Troubleshooting:
1. Check firewall and antivirus settings
2. Test with different network connection
3. Use telnet to test server connectivity
4. Review email client logs for errors
5. Try different email client
6. Contact email provider support

When to Contact Support:
- Server settings confirmed but still not working
- Error messages that don't match common issues
- Business email setup requirements
- Migration from old email system
- Custom domain email configuration

Contact Information:
Phone: 1-800-555-0135 (Email Support)
Email: emailhelp@company.com
Live Chat: Available for email setup assistance
Remote Support: Available for complex configurations'''
        }
    ]
    
    # Create the additional documents
    for doc in additional_docs:
        filepath = os.path.join(base_path, doc['category'], doc['filename'])
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(doc['content'])
        print(f'Created: {filepath}')

if __name__ == '__main__':
    create_documents()
    print("Additional sample documents created successfully!")