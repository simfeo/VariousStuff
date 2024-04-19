//
//  ViewController.m
//  LockMyMac
//
//  Created by Smbat.makiyan on 2/3/17.
//  Copyright © 2017 Smbat.makiyan. All rights reserved.
//

#import "ViewController.h"

#import <objc/runtime.h>
#import <Foundation/Foundation.h>

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    // Do any additional setup after loading the view.
}


- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}


- (IBAction)btnLock:(NSButton *)sender {
    NSBundle *bundle = [NSBundle bundleWithPath:@"/Applications/Utilities/Keychain Access.app/Contents/Resources/Keychain.menu"];
    
    Class principalClass = [bundle principalClass];
    
    id instance = [[principalClass alloc] init];
    
    [instance performSelector:@selector(_lockScreenMenuHit:) withObject:nil];
    
}

- (IBAction)btnLockAndExit:(NSButton *)sender {
    NSBundle *bundle = [NSBundle bundleWithPath:@"/Applications/Utilities/Keychain Access.app/Contents/Resources/Keychain.menu"];
    
    Class principalClass = [bundle principalClass];
    
    id instance = [[principalClass alloc] init];
    
    [instance performSelector:@selector(_lockScreenMenuHit:) withObject:nil];
    
    exit(0);
    
}
@end
